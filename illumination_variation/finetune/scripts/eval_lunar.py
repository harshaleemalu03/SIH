#!/usr/bin/env python3
"""
Comprehensive Evaluation script for EfficientLoFTR on StereoLunar dataset.

Computes:
1. Relative pose estimation: AUC@5°, AUC@10°, AUC@20°, Pose Success @5°, @10°, @20°, Registration Success
2. Direct 3D->2D ground-truth correspondence metrics via warp_kpts:
   - MMA@1, MMA@3, MMA@5, MMA@10 (Mean Matching Accuracy at 1, 3, 5, 10 pixels)
   - Correspondence RMSE (in pixels)
   - Precision@5px (% of predicted matches within 5px of true projected surface)
3. Epipolar precision prec@5e-4 (diagnostic metric)
4. Density & Efficiency: Number of matches, average runtime (ms/pair), peak GPU VRAM (GB)
5. Stratified evaluation & plots across illumination conditions (same vs moderate vs severe sun difference)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import OrderedDict

project_root = Path(__file__).resolve().parent.parent / "EfficientLoFTR"
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config.default import get_cfg_defaults
from src.loftr import LoFTR, reparameter
from src.datasets.stereo_lunar import StereoLunarDataset
from src.utils.misc import lower_config
from src.loftr.utils.geometry import warp_kpts
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    error_auc,
    epidist_prec
)


def compute_pair_correspondence_metrics(mkpts0, mkpts1, depth0, depth1, T_0to1, K0, K1):
    """
    Computes direct 2D correspondence errors using projected ground truth geometry.
    """
    M = len(mkpts0)
    if M == 0:
        return {
            'mma@1': 0.0, 'mma@3': 0.0, 'mma@5': 0.0, 'mma@10': 0.0,
            'rmse_px': np.nan, 'precision@5': 0.0,
            'valid_gt_count': 0, 'corr_errors': np.array([])
        }

    kpts0 = mkpts0.unsqueeze(0)  # [1, M, 2]
    valid_mask, w_kpts0 = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1)
    vmask = valid_mask[0]        # [M]

    valid_count = int(vmask.sum().item())
    if valid_count == 0:
        return {
            'mma@1': 0.0, 'mma@3': 0.0, 'mma@5': 0.0, 'mma@10': 0.0,
            'rmse_px': np.nan, 'precision@5': 0.0,
            'valid_gt_count': 0, 'corr_errors': np.array([])
        }

    pred_pts1 = mkpts1[vmask]
    true_pts1 = w_kpts0[0][vmask]

    # Euclidean reprojection error in pixels
    corr_errs = torch.norm(pred_pts1 - true_pts1, dim=-1).cpu().numpy()

    mma1 = float(np.mean(corr_errs <= 1.0)) * 100.0
    mma3 = float(np.mean(corr_errs <= 3.0)) * 100.0
    mma5 = float(np.mean(corr_errs <= 5.0)) * 100.0
    mma10 = float(np.mean(corr_errs <= 10.0)) * 100.0
    rmse = float(np.sqrt(np.mean(corr_errs ** 2)))
    precision5 = float((corr_errs <= 5.0).sum() / M) * 100.0

    return {
        'mma@1': mma1,
        'mma@3': mma3,
        'mma@5': mma5,
        'mma@10': mma10,
        'rmse_px': rmse,
        'precision@5': precision5,
        'valid_gt_count': valid_count,
        'corr_errors': corr_errs
    }


def evaluate_checkpoint(ckpt_path, config, dataset, device, method_name="Model"):
    print(f"\n[*] Evaluating [{method_name}] with weights from: {ckpt_path or 'Scratch / Random'}")

    _config = lower_config(config)
    matcher = LoFTR(config=_config['loftr'])

    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        matcher.load_state_dict(state_dict, strict=False)
        print(f"[✓] Successfully loaded weights from {ckpt_path}")

    # Reparameterize RepVGG backbone
    matcher = reparameter(matcher)
    matcher = matcher.eval().to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    all_data = {
        'identifiers': [],
        'scores': [],
        'epi_errs': [],
        'R_errs': [],
        't_errs': [],
        'pose_errs': [],
        'num_matches': [],
        'mma@1': [],
        'mma@3': [],
        'mma@5': [],
        'mma@10': [],
        'rmse_px': [],
        'precision@5': [],
        'all_corr_errors': [],
        'pair_corr_errors': [],
        'baselines': [],
        'runtimes': []
    }

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(loader, desc=f"Evaluating {method_name}")):
            batch = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                else:
                    batch[k] = v

            t0 = time.perf_counter()
            matcher(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            pair_time_ms = (t1 - t0) * 1000.0

            # 1. Epipolar and Pose Errors
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, config)

            # 2. Geometric Ground-Truth Correspondence Metrics
            mkpts0 = batch['mkpts0_f']
            mkpts1 = batch['mkpts1_f']
            depth0 = batch['depth0']
            depth1 = batch['depth1']
            T_0to1 = batch['T_0to1']
            K0 = batch['K0']
            K1 = batch['K1']

            corr_metrics = compute_pair_correspondence_metrics(
                mkpts0, mkpts1, depth0, depth1, T_0to1, K0, K1
            )

            rel_pair_names = list(zip(*batch['pair_names']))
            identifier = '#'.join(rel_pair_names[0])
            mconf = batch['mconf']
            score_val = float(batch['score'][0]) if 'score' in batch else 1.0

            # Pose error for this pair
            r_err = float(batch['R_errs'][0][0])
            t_err = float(batch['t_errs'][0][0])
            p_err = max(r_err, t_err)
            gt_t_norm = float(torch.norm(T_0to1[0, :3, 3]).item())

            all_data['identifiers'].append(identifier)
            all_data['scores'].append(score_val)
            all_data['epi_errs'].append(batch['epi_errs'].cpu().numpy())
            all_data['R_errs'].append(r_err)
            all_data['t_errs'].append(t_err)
            all_data['pose_errs'].append(p_err)
            all_data['baselines'].append(gt_t_norm)
            all_data['num_matches'].append(len(mconf))
            all_data['runtimes'].append(pair_time_ms)

            all_data['mma@1'].append(corr_metrics['mma@1'])
            all_data['mma@3'].append(corr_metrics['mma@3'])
            all_data['mma@5'].append(corr_metrics['mma@5'])
            all_data['mma@10'].append(corr_metrics['mma@10'])
            if not np.isnan(corr_metrics['rmse_px']):
                all_data['rmse_px'].append(corr_metrics['rmse_px'])
            all_data['precision@5'].append(corr_metrics['precision@5'])
            all_data['all_corr_errors'].extend(corr_metrics['corr_errors'])
            all_data['pair_corr_errors'].append(np.array(corr_metrics['corr_errors']))

    zero_baseline_thr = getattr(config.TRAINER, 'ZERO_BASELINE_THR', 1.0)
    baselines = np.array(all_data['baselines'])
    stereo_mask = (baselines >= zero_baseline_thr)
    zero_mask = (baselines < zero_baseline_thr)

    # 1. Genuine Stereo Registration (Essential Matrix: 6-DoF R + t)
    n_stereo = int(np.sum(stereo_mask))
    if n_stereo > 0:
        stereo_pose_errs = np.array(all_data['pose_errs'])[stereo_mask]
        stereo_r_errs = np.array(all_data['R_errs'])[stereo_mask]
        stereo_t_errs = np.array(all_data['t_errs'])[stereo_mask]
        stereo_aucs = error_auc(stereo_pose_errs, [5, 10, 20])
        stereo_succ_5 = float(np.mean(stereo_pose_errs <= 5.0)) * 100.0
        stereo_succ_10 = float(np.mean(stereo_pose_errs <= 10.0)) * 100.0
        stereo_succ_20 = float(np.mean(stereo_pose_errs <= 20.0)) * 100.0
        stereo_summary = {
            'count': n_stereo,
            'auc@5': stereo_aucs['auc@5'] * 100.0,
            'auc@10': stereo_aucs['auc@10'] * 100.0,
            'auc@20': stereo_aucs['auc@20'] * 100.0,
            'pose_success@5': stereo_succ_5,
            'pose_success@10': stereo_succ_10,
            'pose_success@20': stereo_succ_20,
            'mean_r_err': float(np.mean(stereo_r_errs)),
            'mean_t_err': float(np.mean(stereo_t_errs)),
            'med_r_err': float(np.median(stereo_r_errs)),
            'med_t_err': float(np.median(stereo_t_errs)),
        }
    else:
        stereo_summary = {
            'count': 0, 'auc@5': 0.0, 'auc@10': 0.0, 'auc@20': 0.0,
            'pose_success@5': 0.0, 'pose_success@10': 0.0, 'pose_success@20': 0.0,
            'mean_r_err': 0.0, 'mean_t_err': 0.0, 'med_r_err': 0.0, 'med_t_err': 0.0
        }

    # 2. Zero-Baseline Registration (Homography: 3-DoF Pure Rotation R, t = 0)
    n_zero = int(np.sum(zero_mask))
    if n_zero > 0:
        zero_r_errs = np.array(all_data['R_errs'])[zero_mask]
        zero_aucs = error_auc(zero_r_errs, [5, 10, 20])
        zero_succ_1 = float(np.mean(zero_r_errs <= 1.0)) * 100.0
        zero_succ_3 = float(np.mean(zero_r_errs <= 3.0)) * 100.0
        zero_succ_5 = float(np.mean(zero_r_errs <= 5.0)) * 100.0
        zero_succ_10 = float(np.mean(zero_r_errs <= 10.0)) * 100.0
        zero_summary = {
            'count': n_zero,
            'auc@5': zero_aucs['auc@5'] * 100.0,
            'auc@10': zero_aucs['auc@10'] * 100.0,
            'auc@20': zero_aucs['auc@20'] * 100.0,
            'rot_success@1': zero_succ_1,
            'rot_success@3': zero_succ_3,
            'rot_success@5': zero_succ_5,
            'rot_success@10': zero_succ_10,
            'mean_r_err': float(np.mean(zero_r_errs)),
            'med_r_err': float(np.median(zero_r_errs)),
        }
    else:
        zero_summary = {
            'count': 0, 'auc@5': 0.0, 'auc@10': 0.0, 'auc@20': 0.0,
            'rot_success@1': 0.0, 'rot_success@3': 0.0, 'rot_success@5': 0.0, 'rot_success@10': 0.0,
            'mean_r_err': 0.0, 'med_r_err': 0.0
        }

    # Epipolar diagnostic
    precs = epidist_prec(np.array(all_data['epi_errs'], dtype=object), [config.TRAINER.EPI_ERR_THR], True)

    # Correspondence Metrics
    mma1 = float(np.mean(all_data['mma@1']))
    mma3 = float(np.mean(all_data['mma@3']))
    mma5 = float(np.mean(all_data['mma@5']))
    mma10 = float(np.mean(all_data['mma@10']))

    # Globally aggregated RMSE across all valid correspondences: sqrt(sum(error^2) / N)
    all_valid_errors = np.concatenate([e for e in all_data['pair_corr_errors'] if len(e) > 0]) if any(len(e) > 0 for e in all_data['pair_corr_errors']) else np.array([])
    if len(all_valid_errors) > 0:
        rmse = float(np.sqrt(np.mean(all_valid_errors ** 2)))
    else:
        rmse = 0.0
    precision5 = float(np.mean(all_data['precision@5']))
    avg_matches = float(np.mean(all_data['num_matches']))
    avg_runtime = float(np.mean(all_data['runtimes']))

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    summary = {
        'method': method_name,
        'stereo': stereo_summary,
        'zero_baseline': zero_summary,
        'mma@1': mma1,
        'mma@3': mma3,
        'mma@5': mma5,
        'mma@10': mma10,
        'rmse_px': rmse,
        'precision@5': precision5,
        'num_matches': avg_matches,
        'runtime_ms': avg_runtime,
        'peak_vram_gb': peak_vram,
        'prec@5e-4': precs[f"prec@{config.TRAINER.EPI_ERR_THR:.0e}"] * 100.0,
        'raw_data': all_data
    }

    return summary


def analyze_illumination(summary):
    raw = summary['raw_data']
    scores = np.array(raw['scores'])
    baselines = np.array(raw['baselines'])
    zero_baseline_thr = 1.0
    stereo_mask = (baselines >= zero_baseline_thr)
    zero_mask = (baselines < zero_baseline_thr)

    unique_scores = sorted(list(set(scores)), reverse=True)
    strata_results = OrderedDict()
    labels = {
        1.0: "Same Illumination (ΔSun = 0°)",
        0.9: "Moderate Sun Difference (~70°)",
        0.8: "Severe Sun Difference (~140°)"
    }

    for sc in unique_scores:
        mask = (scores == sc)
        if np.sum(mask) == 0:
            continue
        label = labels.get(round(sc, 1), f"Score {sc:.1f}")

        # Stereo subset in this stratum
        s_mask = mask & stereo_mask
        s_count = int(np.sum(s_mask))
        if s_count > 0:
            s_pose_errs = np.array(raw['pose_errs'])[s_mask]
            s_aucs = error_auc(s_pose_errs, [5, 10, 20])
            s_auc10 = s_aucs['auc@10'] * 100.0
            s_succ10 = float(np.mean(s_pose_errs <= 10.0)) * 100.0
        else:
            s_auc10, s_succ10 = float('nan'), float('nan')

        # Zero-baseline subset in this stratum
        z_mask = mask & zero_mask
        z_count = int(np.sum(z_mask))
        if z_count > 0:
            z_r_errs = np.array(raw['R_errs'])[z_mask]
            z_aucs = error_auc(z_r_errs, [5, 10, 20])
            z_auc10 = z_aucs['auc@10'] * 100.0
            z_succ1 = float(np.mean(z_r_errs <= 1.0)) * 100.0
        else:
            z_auc10, z_succ1 = float('nan'), float('nan')

        sub_mma1 = float(np.mean(np.array(raw['mma@1'])[mask]))
        sub_mma3 = float(np.mean(np.array(raw['mma@3'])[mask]))
        sub_mma5 = float(np.mean(np.array(raw['mma@5'])[mask]))
        sub_mma10 = float(np.mean(np.array(raw['mma@10'])[mask]))
        sub_prec5 = float(np.mean(np.array(raw['precision@5'])[mask]))
        sub_matches = float(np.mean(np.array(raw['num_matches'])[mask]))
        
        # Globally aggregated RMSE for stratum
        stratum_pair_errs = [raw['pair_corr_errors'][i] for i in range(len(scores)) if mask[i] and len(raw['pair_corr_errors'][i]) > 0]
        if len(stratum_pair_errs) > 0:
            stratum_errs = np.concatenate(stratum_pair_errs)
            sub_rmse = float(np.sqrt(np.mean(stratum_errs ** 2)))
        else:
            sub_rmse = 0.0

        strata_results[label] = {
            'pairs_count': int(np.sum(mask)),
            'stereo_count': s_count,
            'stereo_auc@10': s_auc10,
            'stereo_succ@10': s_succ10,
            'zero_count': z_count,
            'zero_auc@10': z_auc10,
            'zero_succ@1': z_succ1,
            'mma@3': sub_mma3,
            'mma@5': sub_mma5,
            'mma@10': sub_mma10,
            'rmse_px': sub_rmse,
            'precision@5': sub_prec5,
            'num_matches': sub_matches,
        }

    return strata_results


def plot_illumination_analysis(pretrained_sum, finetuned_sum, output_path="eval_illumination_analysis.png"):
    pre_strata = analyze_illumination(pretrained_sum) if pretrained_sum else None
    ft_strata = analyze_illumination(finetuned_sum) if finetuned_sum else None

    conditions = list(ft_strata.keys() if ft_strata else pre_strata.keys())
    if not conditions:
        return

    x = np.arange(len(conditions))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: MMA@3 and MMA@5
    ax1 = axes[0]
    if pre_strata and ft_strata:
        pre_mma5 = [pre_strata[c]['mma@5'] for c in conditions]
        ft_mma5 = [ft_strata[c]['mma@5'] for c in conditions]
        ax1.bar(x - width/2, pre_mma5, width, label='Pretrained', color='#90caf9')
        ax1.bar(x + width/2, ft_mma5, width, label='Fine-Tuned', color='#1976d2')
    elif ft_strata:
        ft_mma5 = [ft_strata[c]['mma@5'] for c in conditions]
        ax1.bar(x, ft_mma5, width, label='Fine-Tuned', color='#1976d2')
    ax1.set_title('Mean Matching Accuracy (MMA@5px) vs. Illumination')
    ax1.set_ylabel('MMA@5 (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.split('(')[0].strip() for c in conditions], rotation=15)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Plot 2: Number of Matches
    ax2 = axes[1]
    if pre_strata and ft_strata:
        pre_matches = [pre_strata[c]['num_matches'] for c in conditions]
        ft_matches = [ft_strata[c]['num_matches'] for c in conditions]
        ax2.bar(x - width/2, pre_matches, width, label='Pretrained', color='#a5d6a7')
        ax2.bar(x + width/2, ft_matches, width, label='Fine-Tuned', color='#388e3c')
    elif ft_strata:
        ft_matches = [ft_strata[c]['num_matches'] for c in conditions]
        ax2.bar(x, ft_matches, width, label='Fine-Tuned', color='#388e3c')
    ax2.set_title('Number of Matches vs. Illumination')
    ax2.set_ylabel('Average # Matches')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.split('(')[0].strip() for c in conditions], rotation=15)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Plot 3: Cumulative Error CDF
    ax3 = axes[2]
    thresholds = np.linspace(0.1, 15.0, 100)
    if pre_strata and len(pretrained_sum['raw_data']['all_corr_errors']) > 0:
        pre_errs = np.array(pretrained_sum['raw_data']['all_corr_errors'])
        pre_cdf = [np.mean(pre_errs <= t) * 100.0 for t in thresholds]
        ax3.plot(thresholds, pre_cdf, label='Pretrained', color='#e53935', linewidth=2)

    if ft_strata and len(finetuned_sum['raw_data']['all_corr_errors']) > 0:
        ft_errs = np.array(finetuned_sum['raw_data']['all_corr_errors'])
        ft_cdf = [np.mean(ft_errs <= t) * 100.0 for t in thresholds]
        ax3.plot(thresholds, ft_cdf, label='Fine-Tuned', color='#1e88e5', linewidth=2)

    ax3.set_title('Cumulative Correspondence Error (CDF)')
    ax3.set_xlabel('Error Threshold (pixels)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.axvline(x=3.0, color='gray', linestyle=':', label='3px threshold')
    ax3.axvline(x=5.0, color='black', linestyle=':', label='5px threshold')
    ax3.legend()
    ax3.grid(linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\n[✓] Illumination analysis plot saved to: {output_path}")


def print_comparison_table(summaries):
    print("\n" + "=" * 125)
    print("               EVALUATION RESULTS: PRETRAINED VS. FINE-TUNED (SEPARATED GEOMETRIES)")
    print("=" * 125)

    # TABLE 1: GENUINE STEREO REGISTRATION (Essential Matrix: 6-DoF R + t)
    print("\n" + "-" * 125)
    print(" TABLE 1: GENUINE STEREO REGISTRATION (Essential Matrix: 6-DoF R + t, Epipolar Geometry)")
    print("-" * 125)
    stereo_header = (
        f"{'Method':<12} | {'Pairs':<5} | {'AUC@5':<6} | {'AUC@10':<6} | {'AUC@20':<6} | "
        f"{'Pose Succ@5°':<13} | {'Pose Succ@10°':<14} | {'Pose Succ@20°':<14} | "
        f"{'Mean R_err':<10} | {'Mean t_err':<10}"
    )
    print(stereo_header)
    print("-" * len(stereo_header))
    for s in summaries:
        st = s['stereo']
        line = (
            f"{s['method']:<12} | {st['count']:<5} | "
            f"{st['auc@5']:>6.2f} | {st['auc@10']:>6.2f} | {st['auc@20']:>6.2f} | "
            f"{st['pose_success@5']:>12.2f}% | {st['pose_success@10']:>13.2f}% | {st['pose_success@20']:>13.2f}% | "
            f"{st['mean_r_err']:>9.2f}° | {st['mean_t_err']:>9.2f}°"
        )
        print(line)
    print("-" * len(stereo_header))

    # TABLE 2: ZERO-BASELINE REGISTRATION (Homography: 3-DoF Pure Rotation R)
    print("\n" + "-" * 125)
    print(" TABLE 2: ZERO-BASELINE REGISTRATION (Homography: 3-DoF Pure Rotation R, Translation t ≡ 0)")
    print("-" * 125)
    zero_header = (
        f"{'Method':<12} | {'Pairs':<5} | {'AUC@5':<6} | {'AUC@10':<6} | {'AUC@20':<6} | "
        f"{'Rot Succ@1°':<12} | {'Rot Succ@3°':<12} | {'Rot Succ@5°':<12} | "
        f"{'Mean R_err':<10} | {'Med R_err':<10}"
    )
    print(zero_header)
    print("-" * len(zero_header))
    for s in summaries:
        zb = s['zero_baseline']
        line = (
            f"{s['method']:<12} | {zb['count']:<5} | "
            f"{zb['auc@5']:>6.2f} | {zb['auc@10']:>6.2f} | {zb['auc@20']:>6.2f} | "
            f"{zb['rot_success@1']:>11.2f}% | {zb['rot_success@3']:>11.2f}% | {zb['rot_success@5']:>11.2f}% | "
            f"{zb['mean_r_err']:>9.2f}° | {zb['med_r_err']:>9.2f}°"
        )
        print(line)
    print("-" * len(zero_header))

    # TABLE 3: CORRESPONDENCE & SYSTEM METRICS (All Pairs)
    print("\n" + "-" * 125)
    print(" TABLE 3: DENSE CORRESPONDENCE & SYSTEM PERFORMANCE (Global Over All Pairs)")
    print("-" * 125)
    corr_header = (
        f"{'Method':<12} | {'MMA@1':<6} | {'MMA@3':<6} | {'MMA@5':<6} | {'MMA@10':<6} | "
        f"{'RMSE(px)':<8} | {'Precision@5':<11} | {'#Matches':<8} | {'Runtime':<9} | {'VRAM':<7} | {'prec@5e-4':<9}"
    )
    print(corr_header)
    print("-" * len(corr_header))
    for s in summaries:
        line = (
            f"{s['method']:<12} | "
            f"{s['mma@1']:>6.2f}%| {s['mma@3']:>6.2f}%| {s['mma@5']:>6.2f}%| {s['mma@10']:>6.2f}%| "
            f"{s['rmse_px']:>8.2f} | {s['precision@5']:>10.2f}% | "
            f"{s['num_matches']:>8.1f} | "
            f"{s['runtime_ms']:>6.1f} ms | {s['peak_vram_gb']:>5.2f} GB | {s['prec@5e-4']:>8.2f}%"
        )
        print(line)
    print("-" * len(corr_header))


def print_illumination_table(summary):
    strata = analyze_illumination(summary)
    print(f"\n--- Illumination Breakdown: {summary['method']} ---")
    header = (
        f"{'Illumination Condition':<35} | {'Total':<5} | {'Stereo':<6} | {'StereoSucc':<10} | "
        f"{'Zero-BL':<7} | {'ZeroRotSucc':<11} | {'MMA@5':<7} | {'RMSE(px)':<8} | {'Precision@5':<11} | {'#Matches':<8}"
    )
    print(header)
    print("-" * len(header))
    for cond, v in strata.items():
        s_succ_str = f"{v['stereo_succ@10']:>6.1f}%" if not np.isnan(v['stereo_succ@10']) else "   N/A "
        z_succ_str = f"{v['zero_succ@1']:>6.1f}%" if not np.isnan(v['zero_succ@1']) else "   N/A "
        print(
            f"{cond:<35} | {v['pairs_count']:<5} | {v['stereo_count']:<6} | {s_succ_str:<10} | "
            f"{v['zero_count']:<7} | {z_succ_str:<11} | "
            f"{v['mma@5']:>6.2f}%| {v['rmse_px']:>8.2f} | "
            f"{v['precision@5']:>10.2f}% | {v['num_matches']:>8.1f}"
        )
    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(description="StereoLunar + EfficientLoFTR Evaluation")
    parser.add_argument('--ckpt', type=str, default=None, help='Fine-tuned model checkpoint path')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Pretrained foundation checkpoint path')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on (test, val, train)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Execution device (auto, cuda, cpu)')
    parser.add_argument('--plot_out', type=str, default="eval_illumination_analysis.png", help='Path to save plot')
    args = parser.parse_args()

    if args.device == 'cuda':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        # Auto: test if CUDA has at least 1.5GB free memory
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                if free_mem >= 1.5:
                    device = torch.device('cuda')
                else:
                    logger.warning(f"CUDA available but only {free_mem:.2f} GB free. Falling back to CPU for evaluation.")
                    device = torch.device('cpu')
            except Exception:
                device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    print(f"[*] Evaluation running on device: {device}")

    config = get_cfg_defaults()
    config.merge_from_file(str(project_root / "configs" / "loftr" / "eloftr_lunar.py"))
    config.merge_from_file(str(project_root / "configs" / "data" / "stereo_lunar_512.py"))

    data_root = project_root / "data" / "stereo_lunar" / "scenes"
    split_file = project_root / "data" / "stereo_lunar" / "split.npz"

    test_dataset = StereoLunarDataset(
        root_dir=str(data_root),
        split_file=str(split_file),
        split=args.split,
        img_resize=512,
    )

    summaries = []
    pre_sum = None
    ft_sum = None

    if args.pretrained_ckpt:
        pre_sum = evaluate_checkpoint(args.pretrained_ckpt, config, test_dataset, device, method_name="Pretrained")
        summaries.append(pre_sum)

    if args.ckpt:
        ft_sum = evaluate_checkpoint(args.ckpt, config, test_dataset, device, method_name="Fine-Tuned")
        summaries.append(ft_sum)

    # 1. Print Master Comparison Table
    print_comparison_table(summaries)

    # 2. Print Illumination Breakdown Tables
    if pre_sum:
        print_illumination_table(pre_sum)
    if ft_sum:
        print_illumination_table(ft_sum)

    # 3. Produce Visualization Plots
    plot_illumination_analysis(pre_sum, ft_sum, output_path=args.plot_out)


if __name__ == '__main__':
    main()
