#!/usr/bin/env python3
"""
benchmark_test_set.py - Comprehensive Benchmark Suite for EfficientLoFTR on StereoLunar

Key Guarantees:
1. Zero Data Leakage:
   - Formally verifies Scene(train) ∩ Scene(test) = ∅ and Scene(val) ∩ Scene(test) = ∅ at startup.
   - Strictly draws evaluation pairs from the held-out test split (split.npz).
   - Ground-truth depth, intrinsics, and poses are strictly isolated for evaluation (never seen by model).
   - Strict evaluation mode: torch.inference_mode(), model.eval(), frozen weights, RepVGG reparameterized.

2. Comprehensive Industry-Grade Metrics:
   - 6-DoF Genuine Stereo Registration (baseline >= 1.0m): AUC@5°, 10°, 20°, Pose Success @5°, 10°, 20°, R_err, t_err.
   - 3-DoF Pure Rotation Zero-Baseline Registration (baseline < 1.0m): AUC@5°, 10°, 20°, Rot Success @1°, 3°, 5°, 10°, R_err.
   - 3D Projected Ground-Truth Surface Correspondence (via warp_kpts):
     MMA@1px, MMA@2px, MMA@3px, MMA@5px, MMA@10px, Global Reprojection RMSE (px), Precision@3px, Precision@5px.
   - Diagnostic: Epipolar precision prec@5e-4.
   - System Efficiency: Matches count, inference latency (ms/pair), throughput (pairs/sec), peak VRAM (GB).
   - Stratified breakdown across illumination deltas (Same ΔSun=0°, Moderate ΔSun≈70°, Severe ΔSun≈140°).

3. Proper Storage of Generated Artifacts & Metrics:
   - Output directory: finetune/benchmark_results/<run_name>/
   - summary_metrics.json : Full structured machine-readable metrics and experiment configuration.
   - per_pair_metrics.csv : Granular per-pair results (pose error, correspondence accuracy, runtime, baseline).
   - benchmark_report.md  : Publication-ready markdown report with formatted tables.
   - reprojection_error_cdf.png : High-res cumulative error distribution curve.
   - illumination_breakdown.png : Stratified performance bar charts.
   - error_distributions.png    : Histograms of angular and reprojection errors.
   - leaderboard.csv     : Central master registry appending headline metrics across all benchmark runs.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# Ensure EfficientLoFTR is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent / "EfficientLoFTR"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


# ==============================================================================
# 1. Zero Data Leakage Verification
# ==============================================================================
def verify_dataset_integrity_and_leakage(split_file_path):
    """
    Formally inspects split.npz to guarantee strict scene-level partitioning.
    Raises AssertionError if any scene overlap is detected.
    """
    logger.info(f"Verifying dataset partitioning integrity from: {split_file_path}")
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Manifest not found: {split_file_path}")

    data = np.load(split_file_path, allow_pickle=True)
    scenes = data['scenes']
    pairs = data['pairs']
    split = data['split']

    train_mask = (split == 'train')
    val_mask = (split == 'val')
    test_mask = (split == 'test')

    train_scene_ids = set(pairs[train_mask][:, 0])
    val_scene_ids = set(pairs[val_mask][:, 0])
    test_scene_ids = set(pairs[test_mask][:, 0])

    overlap_train_test = train_scene_ids.intersection(test_scene_ids)
    overlap_val_test = val_scene_ids.intersection(test_scene_ids)
    overlap_train_val = train_scene_ids.intersection(val_scene_ids)

    if overlap_train_test:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED! {len(overlap_train_test)} scene(s) overlap between train and test splits!"
        )
    if overlap_val_test:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED! {len(overlap_val_test)} scene(s) overlap between val and test splits!"
        )
    if overlap_train_val:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED! {len(overlap_train_val)} scene(s) overlap between train and val splits!"
        )

    logger.info(
        f"[✓] Zero Data Leakage Verified: "
        f"{len(train_scene_ids)} Train Scenes ({np.sum(train_mask)} pairs), "
        f"{len(val_scene_ids)} Val Scenes ({np.sum(val_mask)} pairs), "
        f"{len(test_scene_ids)} Test Scenes ({np.sum(test_mask)} pairs). "
        f"Strictly disjoint: Scenes(train) ∩ Scenes(test) = ∅."
    )

    return {
        'total_scenes': len(scenes),
        'train_scenes_count': len(train_scene_ids),
        'val_scenes_count': len(val_scene_ids),
        'test_scenes_count': len(test_scene_ids),
        'test_pairs_count': int(np.sum(test_mask)),
        'test_scene_names': [str(scenes[int(sid)]) for sid in sorted(test_scene_ids)]
    }


# ==============================================================================
# 2. Model Initialization & Setup
# ==============================================================================
def load_benchmark_model(ckpt_path, config, device, method_name="Model"):
    """
    Loads LoFTR model, loads checkpoint weights (if provided), reparameterizes
    RepVGG backbone, disables gradients, and switches to eval mode.
    """
    _config = lower_config(config)
    matcher = LoFTR(config=_config['loftr'])

    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        logger.info(f"Loading weights for [{method_name}] from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Load weights (LoFTR strips 'matcher.' prefix automatically in load_state_dict)
        msg = matcher.load_state_dict(state_dict, strict=False)
        logger.info(f"Weights loaded successfully for [{method_name}] (missing keys: {len(msg.missing_keys)})")
    else:
        logger.warning(f"No checkpoint provided for [{method_name}]. Initialized from scratch.")

    # Reparameterize RepVGG backbone into single inference convolutions
    matcher = reparameter(matcher)
    matcher = matcher.eval().to(device)

    # Freeze all parameters
    for p in matcher.parameters():
        p.requires_grad = False

    return matcher


# ==============================================================================
# 3. Geometric & Ground-Truth Correspondence Evaluation
# ==============================================================================
def compute_pair_correspondence_metrics(mkpts0, mkpts1, depth0, depth1, T_0to1, K0, K1):
    """
    Computes direct 2D correspondence errors using projected ground truth geometry via warp_kpts.
    """
    M = len(mkpts0)
    if M == 0:
        return {
            'mma@1': 0.0, 'mma@2': 0.0, 'mma@3': 0.0, 'mma@5': 0.0, 'mma@10': 0.0,
            'rmse_px': np.nan, 'precision@3': 0.0, 'precision@5': 0.0,
            'valid_gt_count': 0, 'corr_errors': np.array([])
        }

    kpts0 = mkpts0.unsqueeze(0)  # [1, M, 2]
    valid_mask, w_kpts0 = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1)
    vmask = valid_mask[0]        # [M]

    valid_count = int(vmask.sum().item())
    if valid_count == 0:
        return {
            'mma@1': 0.0, 'mma@2': 0.0, 'mma@3': 0.0, 'mma@5': 0.0, 'mma@10': 0.0,
            'rmse_px': np.nan, 'precision@3': 0.0, 'precision@5': 0.0,
            'valid_gt_count': 0, 'corr_errors': np.array([])
        }

    pred_pts1 = mkpts1[vmask]
    true_pts1 = w_kpts0[0][vmask]

    # Euclidean pixel error
    corr_errs = torch.norm(pred_pts1 - true_pts1, dim=-1).cpu().numpy()

    mma1 = float(np.mean(corr_errs <= 1.0)) * 100.0
    mma2 = float(np.mean(corr_errs <= 2.0)) * 100.0
    mma3 = float(np.mean(corr_errs <= 3.0)) * 100.0
    mma5 = float(np.mean(corr_errs <= 5.0)) * 100.0
    mma10 = float(np.mean(corr_errs <= 10.0)) * 100.0
    rmse = float(np.sqrt(np.mean(corr_errs ** 2)))
    precision3 = float((corr_errs <= 3.0).sum() / M) * 100.0
    precision5 = float((corr_errs <= 5.0).sum() / M) * 100.0

    return {
        'mma@1': mma1,
        'mma@2': mma2,
        'mma@3': mma3,
        'mma@5': mma5,
        'mma@10': mma10,
        'rmse_px': rmse,
        'precision@3': precision3,
        'precision@5': precision5,
        'valid_gt_count': valid_count,
        'corr_errors': corr_errs
    }


# ==============================================================================
# 4. Evaluation Loop on Benchmark Dataset
# ==============================================================================
def evaluate_model_on_dataset(matcher, dataset, config, device, method_name="Model"):
    """
    Executes benchmark inference across test dataset.
    """
    logger.info(f"Running inference for [{method_name}] on {len(dataset)} pair(s)...")

    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    all_data = {
        'identifiers': [],
        'scenes': [],
        'scores': [],
        'baselines': [],
        'is_stereo': [],
        'num_matches': [],
        'R_errs': [],
        't_errs': [],
        'pose_errs': [],
        'epi_errs': [],
        'mma@1': [],
        'mma@2': [],
        'mma@3': [],
        'mma@5': [],
        'mma@10': [],
        'rmse_px': [],
        'precision@3': [],
        'precision@5': [],
        'valid_gt_counts': [],
        'all_corr_errors': [],
        'pair_corr_errors': [],
        'runtimes': []
    }

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda')
    )

    zero_baseline_thr = getattr(config.TRAINER, 'ZERO_BASELINE_THR', 1.0)

    with torch.inference_mode():
        for idx, sample in enumerate(tqdm(loader, desc=f"Benchmarking {method_name}")):
            batch = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                else:
                    batch[k] = v

            # Measure forward matching time
            t0 = time.perf_counter()
            matcher(batch)
            if torch.cuda.is_available() and device.type == 'cuda':
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
            scene_name = str(batch['scene_id'][0]) if 'scene_id' in batch else rel_pair_names[0][0].split('/')[0]
            mconf = batch['mconf']
            score_val = float(batch['score'][0]) if 'score' in batch else 1.0

            r_err = float(batch['R_errs'][0][0])
            t_err = float(batch['t_errs'][0][0])
            p_err = max(r_err, t_err)
            gt_t_norm = float(torch.norm(T_0to1[0, :3, 3]).item())
            is_stereo = bool(gt_t_norm >= zero_baseline_thr)

            all_data['identifiers'].append(identifier)
            all_data['scenes'].append(scene_name)
            all_data['scores'].append(score_val)
            all_data['baselines'].append(gt_t_norm)
            all_data['is_stereo'].append(is_stereo)
            all_data['num_matches'].append(len(mconf))
            all_data['runtimes'].append(pair_time_ms)
            all_data['R_errs'].append(r_err)
            all_data['t_errs'].append(t_err)
            all_data['pose_errs'].append(p_err)
            all_data['epi_errs'].append(batch['epi_errs'].cpu().numpy())

            all_data['mma@1'].append(corr_metrics['mma@1'])
            all_data['mma@2'].append(corr_metrics['mma@2'])
            all_data['mma@3'].append(corr_metrics['mma@3'])
            all_data['mma@5'].append(corr_metrics['mma@5'])
            all_data['mma@10'].append(corr_metrics['mma@10'])
            all_data['rmse_px'].append(corr_metrics['rmse_px'])
            all_data['precision@3'].append(corr_metrics['precision@3'])
            all_data['precision@5'].append(corr_metrics['precision@5'])
            all_data['valid_gt_counts'].append(corr_metrics['valid_gt_count'])
            all_data['all_corr_errors'].extend(corr_metrics['corr_errors'])
            all_data['pair_corr_errors'].append(np.array(corr_metrics['corr_errors']))

    peak_vram = 0.0
    if torch.cuda.is_available() and device.type == 'cuda':
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    return all_data, peak_vram


# ==============================================================================
# 5. Metric Aggregation & Illumination Stratification
# ==============================================================================
def aggregate_benchmark_results(all_data, peak_vram, config, method_name="Model", ckpt_path=None):
    """
    Computes global statistics, partitioned 6-DoF stereo vs 3-DoF zero-baseline metrics,
    correspondence metrics, and stratified illumination breakdowns.
    """
    baselines = np.array(all_data['baselines'])
    zero_baseline_thr = getattr(config.TRAINER, 'ZERO_BASELINE_THR', 1.0)
    stereo_mask = (baselines >= zero_baseline_thr)
    zero_mask = (baselines < zero_baseline_thr)

    # --- 1. Genuine Stereo Registration (6-DoF Essential Matrix) ---
    n_stereo = int(np.sum(stereo_mask))
    if n_stereo > 0:
        stereo_pose_errs = np.array(all_data['pose_errs'])[stereo_mask]
        stereo_r_errs = np.array(all_data['R_errs'])[stereo_mask]
        stereo_t_errs = np.array(all_data['t_errs'])[stereo_mask]
        stereo_aucs = error_auc(stereo_pose_errs, [5, 10, 20])
        stereo_summary = {
            'count': n_stereo,
            'auc@5': float(stereo_aucs['auc@5'] * 100.0),
            'auc@10': float(stereo_aucs['auc@10'] * 100.0),
            'auc@20': float(stereo_aucs['auc@20'] * 100.0),
            'pose_success@5': float(np.mean(stereo_pose_errs <= 5.0) * 100.0),
            'pose_success@10': float(np.mean(stereo_pose_errs <= 10.0) * 100.0),
            'pose_success@20': float(np.mean(stereo_pose_errs <= 20.0) * 100.0),
            'mean_r_err': float(np.mean(stereo_r_errs)),
            'med_r_err': float(np.median(stereo_r_errs)),
            'mean_t_err': float(np.mean(stereo_t_errs)),
            'med_t_err': float(np.median(stereo_t_errs)),
        }
    else:
        stereo_summary = {
            'count': 0, 'auc@5': 0.0, 'auc@10': 0.0, 'auc@20': 0.0,
            'pose_success@5': 0.0, 'pose_success@10': 0.0, 'pose_success@20': 0.0,
            'mean_r_err': 0.0, 'med_r_err': 0.0, 'mean_t_err': 0.0, 'med_t_err': 0.0
        }

    # --- 2. Zero-Baseline Registration (3-DoF Pure Rotation) ---
    n_zero = int(np.sum(zero_mask))
    if n_zero > 0:
        zero_r_errs = np.array(all_data['R_errs'])[zero_mask]
        zero_aucs = error_auc(zero_r_errs, [5, 10, 20])
        zero_summary = {
            'count': n_zero,
            'auc@5': float(zero_aucs['auc@5'] * 100.0),
            'auc@10': float(zero_aucs['auc@10'] * 100.0),
            'auc@20': float(zero_aucs['auc@20'] * 100.0),
            'rot_success@1': float(np.mean(zero_r_errs <= 1.0) * 100.0),
            'rot_success@3': float(np.mean(zero_r_errs <= 3.0) * 100.0),
            'rot_success@5': float(np.mean(zero_r_errs <= 5.0) * 100.0),
            'rot_success@10': float(np.mean(zero_r_errs <= 10.0) * 100.0),
            'mean_r_err': float(np.mean(zero_r_errs)),
            'med_r_err': float(np.median(zero_r_errs)),
        }
    else:
        zero_summary = {
            'count': 0, 'auc@5': 0.0, 'auc@10': 0.0, 'auc@20': 0.0,
            'rot_success@1': 0.0, 'rot_success@3': 0.0, 'rot_success@5': 0.0, 'rot_success@10': 0.0,
            'mean_r_err': 0.0, 'med_r_err': 0.0
        }

    # --- 3. Epipolar Precision Diagnostic ---
    precs = epidist_prec(np.array(all_data['epi_errs'], dtype=object), [config.TRAINER.EPI_ERR_THR], True)
    epipolar_prec = float(precs[f"prec@{config.TRAINER.EPI_ERR_THR:.0e}"] * 100.0)

    # --- 4. Direct 3D Surface Correspondences (Global Over All Valid Pairs) ---
    mma1 = float(np.mean(all_data['mma@1']))
    mma2 = float(np.mean(all_data['mma@2']))
    mma3 = float(np.mean(all_data['mma@3']))
    mma5 = float(np.mean(all_data['mma@5']))
    mma10 = float(np.mean(all_data['mma@10']))

    all_valid_errors = np.concatenate([e for e in all_data['pair_corr_errors'] if len(e) > 0]) if any(len(e) > 0 for e in all_data['pair_corr_errors']) else np.array([])
    if len(all_valid_errors) > 0:
        global_rmse = float(np.sqrt(np.mean(all_valid_errors ** 2)))
    else:
        global_rmse = 0.0

    precision3 = float(np.mean(all_data['precision@3']))
    precision5 = float(np.mean(all_data['precision@5']))
    avg_matches = float(np.mean(all_data['num_matches']))
    avg_runtime_ms = float(np.mean(all_data['runtimes']))
    throughput_fps = float(1000.0 / avg_runtime_ms) if avg_runtime_ms > 0 else 0.0

    # --- 5. Illumination Stratification ---
    scores = np.array(all_data['scores'])
    unique_scores = sorted(list(set(scores)), reverse=True)
    labels = {
        1.0: "Same Illumination (ΔSun = 0°)",
        0.9: "Moderate Sun Difference (~70°)",
        0.8: "Severe Sun Difference (~140°)"
    }
    illumination_strata = OrderedDict()

    for sc in unique_scores:
        mask = (scores == sc)
        if np.sum(mask) == 0:
            continue
        label = labels.get(round(sc, 1), f"Score {sc:.1f}")

        s_mask = mask & stereo_mask
        s_count = int(np.sum(s_mask))
        if s_count > 0:
            s_pose_errs = np.array(all_data['pose_errs'])[s_mask]
            s_aucs = error_auc(s_pose_errs, [5, 10, 20])
            s_auc10 = float(s_aucs['auc@10'] * 100.0)
            s_succ10 = float(np.mean(s_pose_errs <= 10.0) * 100.0)
        else:
            s_auc10, s_succ10 = float('nan'), float('nan')

        z_mask = mask & zero_mask
        z_count = int(np.sum(z_mask))
        if z_count > 0:
            z_r_errs = np.array(all_data['R_errs'])[z_mask]
            z_aucs = error_auc(z_r_errs, [5, 10, 20])
            z_auc10 = float(z_aucs['auc@10'] * 100.0)
            z_succ1 = float(np.mean(z_r_errs <= 1.0) * 100.0)
        else:
            z_auc10, z_succ1 = float('nan'), float('nan')

        sub_mma3 = float(np.mean(np.array(all_data['mma@3'])[mask]))
        sub_mma5 = float(np.mean(np.array(all_data['mma@5'])[mask]))
        sub_mma10 = float(np.mean(np.array(all_data['mma@10'])[mask]))
        sub_prec5 = float(np.mean(np.array(all_data['precision@5'])[mask]))
        sub_matches = float(np.mean(np.array(all_data['num_matches'])[mask]))

        stratum_pair_errs = [all_data['pair_corr_errors'][i] for i in range(len(scores)) if mask[i] and len(all_data['pair_corr_errors'][i]) > 0]
        if len(stratum_pair_errs) > 0:
            stratum_errs = np.concatenate(stratum_pair_errs)
            sub_rmse = float(np.sqrt(np.mean(stratum_errs ** 2)))
        else:
            sub_rmse = 0.0

        illumination_strata[label] = {
            'score': float(sc),
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

    summary = {
        'method': method_name,
        'checkpoint_path': str(ckpt_path) if ckpt_path else "scratch",
        'timestamp': datetime.now().isoformat(),
        'total_evaluated_pairs': len(all_data['identifiers']),
        'stereo': stereo_summary,
        'zero_baseline': zero_summary,
        'mma@1': mma1,
        'mma@2': mma2,
        'mma@3': mma3,
        'mma@5': mma5,
        'mma@10': mma10,
        'rmse_px': global_rmse,
        'precision@3': precision3,
        'precision@5': precision5,
        'num_matches': avg_matches,
        'runtime_ms': avg_runtime_ms,
        'throughput_fps': throughput_fps,
        'peak_vram_gb': peak_vram,
        'epipolar_prec@5e-4': epipolar_prec,
        'illumination_strata': illumination_strata,
        'raw_data': all_data
    }

    return summary


# ==============================================================================
# 6. Artifact Generation & Storage (CSV, JSON, Markdown, Plots, Leaderboard)
# ==============================================================================
def save_benchmark_artifacts(summary, output_dir, baseline_summary=None):
    """
    Saves all metrics and artifacts in a structured folder:
    - summary_metrics.json
    - per_pair_metrics.csv
    - benchmark_report.md
    - reprojection_error_cdf.png
    - illumination_breakdown.png
    - error_distributions.png
    - Appends headline metrics to benchmark_results/leaderboard.csv
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving benchmark artifacts to: {out_path}")

    # 1. Summary JSON (exclude giant raw arrays to keep JSON clean)
    json_summary = {k: v for k, v in summary.items() if k != 'raw_data'}
    with open(out_path / "summary_metrics.json", "w") as f:
        json.dump(json_summary, f, indent=2)

    # 2. Per-Pair Metrics CSV
    raw = summary['raw_data']
    csv_file = out_path / "per_pair_metrics.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_idx", "identifier", "scene", "score", "baseline_m", "is_stereo",
            "num_matches", "r_err_deg", "t_err_deg", "pose_err_deg",
            "mma1", "mma2", "mma3", "mma5", "mma10", "rmse_px", "precision3", "precision5",
            "runtime_ms"
        ])
        for i in range(len(raw['identifiers'])):
            writer.writerow([
                i,
                raw['identifiers'][i],
                raw['scenes'][i],
                raw['scores'][i],
                round(raw['baselines'][i], 4),
                raw['is_stereo'][i],
                raw['num_matches'][i],
                round(raw['R_errs'][i], 4),
                round(raw['t_errs'][i], 4),
                round(raw['pose_errs'][i], 4),
                round(raw['mma@1'][i], 2),
                round(raw['mma@2'][i], 2),
                round(raw['mma@3'][i], 2),
                round(raw['mma@5'][i], 2),
                round(raw['mma@10'][i], 2),
                round(raw['rmse_px'][i], 4) if not np.isnan(raw['rmse_px'][i]) else "NaN",
                round(raw['precision@3'][i], 2),
                round(raw['precision@5'][i], 2),
                round(raw['runtimes'][i], 2)
            ])

    # 3. Benchmark Markdown Report
    report_file = out_path / "benchmark_report.md"
    generate_markdown_report(summary, report_file, baseline_summary)

    # 4. Plots
    generate_benchmark_plots(summary, out_path, baseline_summary)

    # 5. Master Leaderboard Registration
    register_to_leaderboard(summary, out_path.parent / "leaderboard.csv")

    logger.info(f"[✓] All benchmark artifacts saved successfully in {out_path}")


def generate_markdown_report(summary, output_path, baseline_summary=None):
    """
    Generates a GitHub-flavored Markdown report of benchmark results.
    """
    st = summary['stereo']
    zb = summary['zero_baseline']

    md = []
    md.append(f"# EfficientLoFTR Benchmark Report: {summary['method']}")
    md.append(f"\n- **Execution Timestamp**: {summary['timestamp']}")
    md.append(f"- **Checkpoint**: `{summary['checkpoint_path']}`")
    md.append(f"- **Total Evaluated Pairs**: {summary['total_evaluated_pairs']}")
    md.append(f"- **Hardware Allocation**: Peak VRAM {summary['peak_vram_gb']:.2f} GB | Latency {summary['runtime_ms']:.1f} ms/pair ({summary['throughput_fps']:.1f} fps)")

    md.append("\n## 1. Primary Registration Benchmark (Separated Geometries)")
    md.append("\n### Genuine Stereo Pairs (Essential Matrix 6-DoF $R+\\mathbf{t}$, baseline $\\ge 1.0\\text{ m}$)")
    md.append(
        "| Metric | Value |\n| :--- | :---: |\n"
        f"| Evaluated Pairs | {st['count']} |\n"
        f"| AUC @ 5° / 10° / 20° | {st['auc@5']:.2f} / {st['auc@10']:.2f} / {st['auc@20']:.2f} |\n"
        f"| Pose Success @ 5° / 10° / 20° | {st['pose_success@5']:.2f}% / {st['pose_success@10']:.2f}% / {st['pose_success@20']:.2f}% |\n"
        f"| Mean Rotation Error | {st['mean_r_err']:.2f}° (Median: {st['med_r_err']:.2f}°) |\n"
        f"| Mean Translation Error | {st['mean_t_err']:.2f}° (Median: {st['med_t_err']:.2f}°) |"
    )

    md.append("\n### Zero-Baseline Pairs (Homography 3-DoF Pure Rotation $R$, $\\mathbf{t} \\equiv \\mathbf{0}$)")
    md.append(
        "| Metric | Value |\n| :--- | :---: |\n"
        f"| Evaluated Pairs | {zb['count']} |\n"
        f"| AUC @ 5° / 10° / 20° | {zb['auc@5']:.2f} / {zb['auc@10']:.2f} / {zb['auc@20']:.2f} |\n"
        f"| Rotation Success @ 1° / 3° / 5° / 10° | {zb['rot_success@1']:.2f}% / {zb['rot_success@3']:.2f}% / {zb['rot_success@5']:.2f}% / {zb['rot_success@10']:.2f}% |\n"
        f"| Mean Rotation Error | {zb['mean_r_err']:.2f}° (Median: {zb['med_r_err']:.2f}°) |"
    )

    md.append("\n## 2. Direct 3D Surface Correspondence Accuracy (Projected Ground Truth)")
    md.append(
        "| Metric | Value |\n| :--- | :---: |\n"
        f"| Mean Matching Accuracy (MMA@1px) | {summary['mma@1']:.2f}% |\n"
        f"| Mean Matching Accuracy (MMA@3px) | {summary['mma@3']:.2f}% |\n"
        f"| Mean Matching Accuracy (MMA@5px) | {summary['mma@5']:.2f}% |\n"
        f"| Mean Matching Accuracy (MMA@10px) | {summary['mma@10']:.2f}% |\n"
        f"| Global Reprojection RMSE | {summary['rmse_px']:.2f} pixels |\n"
        f"| Precision@5px | {summary['precision@5']:.2f}% |\n"
        f"| Average # Matches per Pair | {summary['num_matches']:.1f} |\n"
        f"| Epipolar Precision (prec@5e-4) | {summary['epipolar_prec@5e-4']:.2f}% |"
    )

    md.append("\n## 3. Illumination Stratification Breakdown")
    md.append(
        "| Illumination Condition | Pairs | Stereo Succ (<=10°) | Zero-BL Rot Succ (<=1°) | MMA@5 | RMSE (px) | Precision@5 | #Matches |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for cond, v in summary['illumination_strata'].items():
        s_succ = f"{v['stereo_succ@10']:.1f}%" if not np.isnan(v['stereo_succ@10']) else "N/A"
        z_succ = f"{v['zero_succ@1']:.1f}%" if not np.isnan(v['zero_succ@1']) else "N/A"
        md.append(
            f"| {cond} | {v['pairs_count']} | {s_succ} | {z_succ} | "
            f"{v['mma@5']:.2f}% | {v['rmse_px']:.2f} | {v['precision@5']:.2f}% | {v['num_matches']:.1f} |"
        )

    if baseline_summary:
        md.append("\n## 4. Side-by-Side Comparison vs. Baseline")
        b_st = baseline_summary['stereo']
        b_zb = baseline_summary['zero_baseline']
        md.append(
            "| Metric | Baseline | Fine-Tuned | Delta / Gain |\n"
            "| :--- | :---: | :---: | :---: |\n"
            f"| Stereo AUC@10 | {b_st['auc@10']:.2f} | {st['auc@10']:.2f} | {st['auc@10'] - b_st['auc@10']:+.2f} |\n"
            f"| Zero-BL Rot Succ@1° | {b_zb['rot_success@1']:.2f}% | {zb['rot_success@1']:.2f}% | {zb['rot_success@1'] - b_zb['rot_success@1']:+.2f}% |\n"
            f"| Zero-BL Mean R_err | {b_zb['mean_r_err']:.2f}° | {zb['mean_r_err']:.2f}° | {zb['mean_r_err'] - b_zb['mean_r_err']:+.2f}° |\n"
            f"| MMA@5px | {baseline_summary['mma@5']:.2f}% | {summary['mma@5']:.2f}% | {summary['mma@5'] - baseline_summary['mma@5']:+.2f}% |\n"
            f"| Correspondence RMSE | {baseline_summary['rmse_px']:.2f} px | {summary['rmse_px']:.2f} px | {summary['rmse_px'] - baseline_summary['rmse_px']:+.2f} px |\n"
            f"| Precision@5px | {baseline_summary['precision@5']:.2f}% | {summary['precision@5']:.2f}% | {summary['precision@5'] - baseline_summary['precision@5']:+.2f}% |\n"
            f"| #Matches | {baseline_summary['num_matches']:.1f} | {summary['num_matches']:.1f} | {summary['num_matches'] - baseline_summary['num_matches']:+.1f} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(md))


def generate_benchmark_plots(summary, output_dir, baseline_summary=None):
    """
    Generates high-resolution visualization figures:
    - reprojection_error_cdf.png
    - illumination_breakdown.png
    - error_distributions.png
    """
    out_path = Path(output_dir)

    # 1. Reprojection Error Cumulative Distribution Function (CDF)
    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = np.linspace(0.1, 15.0, 150)
    ft_errs = np.array(summary['raw_data']['all_corr_errors'])

    if len(ft_errs) > 0:
        ft_cdf = [np.mean(ft_errs <= t) * 100.0 for t in thresholds]
        ax.plot(thresholds, ft_cdf, label=f"{summary['method']}", color='#1976d2', linewidth=2.2)

    if baseline_summary and len(baseline_summary['raw_data']['all_corr_errors']) > 0:
        base_errs = np.array(baseline_summary['raw_data']['all_corr_errors'])
        base_cdf = [np.mean(base_errs <= t) * 100.0 for t in thresholds]
        ax.plot(thresholds, base_cdf, label=f"{baseline_summary['method']}", color='#d32f2f', linewidth=2.0, linestyle='--')

    ax.axvline(x=3.0, color='#757575', linestyle=':', label='3px threshold')
    ax.axvline(x=5.0, color='#212121', linestyle=':', label='5px threshold')
    ax.set_title("Direct 3D Surface Correspondence Accuracy (Reprojection Error CDF)")
    ax.set_xlabel("Reprojection Error Threshold (pixels)")
    ax.set_ylabel("Accuracy (% Matches <= Threshold)")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path / "reprojection_error_cdf.png", dpi=200)
    plt.close()

    # 2. Illumination Breakdown Bar Chart
    ft_strata = summary['illumination_strata']
    conditions = list(ft_strata.keys())
    if conditions:
        x = np.arange(len(conditions))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: MMA@5px vs Illumination
        ax1 = axes[0]
        ft_mma5 = [ft_strata[c]['mma@5'] for c in conditions]
        if baseline_summary:
            base_strata = baseline_summary['illumination_strata']
            base_mma5 = [base_strata[c]['mma@5'] for c in conditions]
            ax1.bar(x - width/2, base_mma5, width, label=baseline_summary['method'], color='#ef9a9a')
            ax1.bar(x + width/2, ft_mma5, width, label=summary['method'], color='#1976d2')
        else:
            ax1.bar(x, ft_mma5, width, label=summary['method'], color='#1976d2')
        ax1.set_title("MMA@5px vs. Sun Illumination Difference")
        ax1.set_ylabel("MMA@5 (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.split('(')[0].strip() for c in conditions], rotation=15)
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        ax1.legend()

        # Subplot 2: Number of Matches vs Illumination
        ax2 = axes[1]
        ft_matches = [ft_strata[c]['num_matches'] for c in conditions]
        if baseline_summary:
            base_matches = [base_strata[c]['num_matches'] for c in conditions]
            ax2.bar(x - width/2, base_matches, width, label=baseline_summary['method'], color='#ffcc80')
            ax2.bar(x + width/2, ft_matches, width, label=summary['method'], color='#388e3c')
        else:
            ax2.bar(x, ft_matches, width, label=summary['method'], color='#388e3c')
        ax2.set_title("Predicted Correspondences vs. Sun Illumination Difference")
        ax2.set_ylabel("Average # Matches")
        ax2.set_xticks(x)
        ax2.set_xticklabels([c.split('(')[0].strip() for c in conditions], rotation=15)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(out_path / "illumination_breakdown.png", dpi=200)
        plt.close()

    # 3. Error Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    raw = summary['raw_data']

    # Angular Rotation Error distribution
    r_errs = [r for r in raw['R_errs'] if not np.isinf(r)]
    ax_r = axes[0]
    ax_r.hist(r_errs, bins=30, range=(0, 20), color='#0288d1', edgecolor='black', alpha=0.7)
    ax_r.set_title("Rotation Error Distribution (capped at 20°)")
    ax_r.set_xlabel("Rotation Error (°)")
    ax_r.set_ylabel("Pair Count")
    ax_r.grid(True, linestyle='--', alpha=0.6)

    # Correspondence Error distribution
    corr_errs_sample = raw['all_corr_errors'][:10000] if len(raw['all_corr_errors']) > 0 else []
    ax_c = axes[1]
    if len(corr_errs_sample) > 0:
        ax_c.hist(corr_errs_sample, bins=30, range=(0, 15), color='#7b1fa2', edgecolor='black', alpha=0.7)
        ax_c.set_title("Correspondence Reprojection Error Distribution (capped at 15px)")
        ax_c.set_xlabel("Reprojection Error (pixels)")
        ax_c.set_ylabel("Keypoint Count")
        ax_c.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path / "error_distributions.png", dpi=200)
    plt.close()


def register_to_leaderboard(summary, leaderboard_csv_path):
    """
    Appends headline metrics to a central leaderboard.csv for multi-model tracking.
    """
    file_exists = os.path.exists(leaderboard_csv_path)
    st = summary['stereo']
    zb = summary['zero_baseline']

    fieldnames = [
        "timestamp", "method", "checkpoint", "pairs",
        "stereo_auc10", "stereo_succ10", "stereo_mean_r", "stereo_mean_t",
        "zero_auc10", "zero_succ1", "zero_mean_r",
        "mma1", "mma3", "mma5", "mma10", "rmse_px", "precision5",
        "matches", "latency_ms", "throughput_fps", "vram_gb"
    ]

    with open(leaderboard_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": summary['timestamp'],
            "method": summary['method'],
            "checkpoint": summary['checkpoint_path'],
            "pairs": summary['total_evaluated_pairs'],
            "stereo_auc10": round(st['auc@10'], 2),
            "stereo_succ10": round(st['pose_success@10'], 2),
            "stereo_mean_r": round(st['mean_r_err'], 2),
            "stereo_mean_t": round(st['mean_t_err'], 2),
            "zero_auc10": round(zb['auc@10'], 2),
            "zero_succ1": round(zb['rot_success@1'], 2),
            "zero_mean_r": round(zb['mean_r_err'], 2),
            "mma1": round(summary['mma@1'], 2),
            "mma3": round(summary['mma@3'], 2),
            "mma5": round(summary['mma@5'], 2),
            "mma10": round(summary['mma@10'], 2),
            "rmse_px": round(summary['rmse_px'], 2),
            "precision5": round(summary['precision@5'], 2),
            "matches": round(summary['num_matches'], 1),
            "latency_ms": round(summary['runtime_ms'], 1),
            "throughput_fps": round(summary['throughput_fps'], 1),
            "vram_gb": round(summary['peak_vram_gb'], 2)
        })

    logger.info(f"[✓] Appended results to master leaderboard: {leaderboard_csv_path}")


# ==============================================================================
# 7. Pretty Terminal Printing
# ==============================================================================
def print_terminal_tables(summaries):
    print("\n" + "=" * 125)
    print("                      STERELUNAR TEST SET BENCHMARK RESULTS")
    print("=" * 125)

    # Table 1: Stereo Registration
    print("\n" + "-" * 125)
    print(" TABLE 1: GENUINE STEREO REGISTRATION (Essential Matrix: 6-DoF R + t, Epipolar Geometry, Baseline >= 1.0m)")
    print("-" * 125)
    s_hdr = (
        f"{'Method':<16} | {'Pairs':<5} | {'AUC@5':<6} | {'AUC@10':<6} | {'AUC@20':<6} | "
        f"{'Pose Succ@5°':<13} | {'Pose Succ@10°':<14} | {'Pose Succ@20°':<14} | "
        f"{'Mean R_err':<10} | {'Mean t_err':<10}"
    )
    print(s_hdr)
    print("-" * len(s_hdr))
    for s in summaries:
        st = s['stereo']
        line = (
            f"{s['method']:<16} | {st['count']:<5} | "
            f"{st['auc@5']:>6.2f} | {st['auc@10']:>6.2f} | {st['auc@20']:>6.2f} | "
            f"{st['pose_success@5']:>12.2f}% | {st['pose_success@10']:>13.2f}% | {st['pose_success@20']:>13.2f}% | "
            f"{st['mean_r_err']:>9.2f}° | {st['mean_t_err']:>9.2f}°"
        )
        print(line)
    print("-" * len(s_hdr))

    # Table 2: Zero-Baseline Registration
    print("\n" + "-" * 125)
    print(" TABLE 2: ZERO-BASELINE REGISTRATION (Homography: 3-DoF Pure Rotation R, Translation t ≡ 0, Baseline < 1.0m)")
    print("-" * 125)
    z_hdr = (
        f"{'Method':<16} | {'Pairs':<5} | {'AUC@5':<6} | {'AUC@10':<6} | {'AUC@20':<6} | "
        f"{'Rot Succ@1°':<12} | {'Rot Succ@3°':<12} | {'Rot Succ@5°':<12} | "
        f"{'Mean R_err':<10} | {'Med R_err':<10}"
    )
    print(z_hdr)
    print("-" * len(z_hdr))
    for s in summaries:
        zb = s['zero_baseline']
        line = (
            f"{s['method']:<16} | {zb['count']:<5} | "
            f"{zb['auc@5']:>6.2f} | {zb['auc@10']:>6.2f} | {zb['auc@20']:>6.2f} | "
            f"{zb['rot_success@1']:>11.2f}% | {zb['rot_success@3']:>11.2f}% | {zb['rot_success@5']:>11.2f}% | "
            f"{zb['mean_r_err']:>9.2f}° | {zb['med_r_err']:>9.2f}°"
        )
        print(line)
    print("-" * len(z_hdr))

    # Table 3: Correspondence & System
    print("\n" + "-" * 125)
    print(" TABLE 3: DENSE 3D SURFACE CORRESPONDENCE & EFFICIENCY (Global Over All Valid Correspondences)")
    print("-" * 125)
    c_hdr = (
        f"{'Method':<16} | {'MMA@1':<6} | {'MMA@3':<6} | {'MMA@5':<6} | {'MMA@10':<6} | "
        f"{'RMSE(px)':<8} | {'Precision@5':<11} | {'#Matches':<8} | {'Runtime':<9} | {'Throughput':<10} | {'Peak VRAM':<9}"
    )
    print(c_hdr)
    print("-" * len(c_hdr))
    for s in summaries:
        line = (
            f"{s['method']:<16} | "
            f"{s['mma@1']:>6.2f}%| {s['mma@3']:>6.2f}%| {s['mma@5']:>6.2f}%| {s['mma@10']:>6.2f}%| "
            f"{s['rmse_px']:>8.2f} | {s['precision@5']:>10.2f}% | "
            f"{s['num_matches']:>8.1f} | "
            f"{s['runtime_ms']:>6.1f} ms | {s['throughput_fps']:>7.1f} fps | {s['peak_vram_gb']:>7.2f} GB"
        )
        print(line)
    print("-" * len(c_hdr))


# ==============================================================================
# 8. Main CLI Function
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark EfficientLoFTR against the StereoLunar Full Test Set (Zero Data Leakage)"
    )
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to fine-tuned model checkpoint (.ckpt)')
    parser.add_argument('--method_name', type=str, default='Fine-Tuned',
                        help='Display name for the evaluated model')
    parser.add_argument('--baseline_ckpt', type=str, default=None,
                        help='Optional baseline checkpoint to evaluate and compare side-by-side (e.g. weights/eloftr_outdoor.ckpt)')
    parser.add_argument('--baseline_name', type=str, default='Pretrained Baseline',
                        help='Display name for the baseline model')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'],
                        help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Optional limit on the number of evaluated pairs (default: all pairs)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Sample every S-th pair across the test set for fast uniform coverage (default: 1)')
    parser.add_argument('--scene', type=str, default=None,
                        help='Optional: evaluate only a specific test scene name')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Execution device (auto, cuda, cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom directory to save benchmark artifacts (default: finetune/benchmark_results/<name>_<timestamp>)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine execution device
    if args.device == 'cuda':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                if free_mem >= 1.2:
                    device = torch.device('cuda')
                else:
                    logger.warning(f"CUDA device detected but only {free_mem:.2f} GB free. Falling back to CPU.")
                    device = torch.device('cpu')
            except Exception:
                device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load configurations
    config = get_cfg_defaults()
    loftr_cfg_path = PROJECT_ROOT / "configs" / "loftr" / "eloftr_lunar.py"
    data_cfg_path = PROJECT_ROOT / "configs" / "data" / "stereo_lunar_512.py"
    config.merge_from_file(str(loftr_cfg_path))
    config.merge_from_file(str(data_cfg_path))

    data_root = PROJECT_ROOT / "data" / "stereo_lunar" / "scenes"
    split_file = PROJECT_ROOT / "data" / "stereo_lunar" / "split.npz"

    # 1. Zero Data Leakage Verification
    leakage_report = verify_dataset_integrity_and_leakage(str(split_file))

    # 2. Instantiate Held-out Test Dataset
    dataset = StereoLunarDataset(
        root_dir=str(data_root),
        split_file=str(split_file),
        scene_name=args.scene,
        split=args.split,
        img_resize=512,
    )

    # Apply stride or max_pairs slicing if requested
    if args.stride > 1 or args.max_pairs is not None:
        original_count = len(dataset.pairs)
        indices = np.arange(0, original_count, args.stride)
        if args.max_pairs is not None and args.max_pairs < len(indices):
            indices = indices[:args.max_pairs]
        dataset.pairs = dataset.pairs[indices]
        logger.info(f"Subsampled test set: {len(dataset)} pairs (from {original_count} pairs via stride={args.stride}, max_pairs={args.max_pairs})")

    # Verify again that no training scenes are in the instantiated test pairs
    train_scenes = set(np.load(str(split_file), allow_pickle=True)['pairs'][np.load(str(split_file), allow_pickle=True)['split'] == 'train'][:, 0])
    instantiated_scenes = set(dataset.pairs[:, 0])
    assert train_scenes.isdisjoint(instantiated_scenes), "FATAL: Test dataset contains scenes from train set!"

    # 3. Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_method = args.method_name.lower().replace(" ", "_").replace("-", "_")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT.parent / "benchmark_results" / f"{clean_method}_{args.split}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    baseline_summary = None

    # 4. Optional Baseline Benchmark
    if args.baseline_ckpt:
        logger.info(f"--- Step 1/2: Benchmarking Baseline [{args.baseline_name}] ---")
        base_matcher = load_benchmark_model(args.baseline_ckpt, config, device, method_name=args.baseline_name)
        base_raw, base_vram = evaluate_model_on_dataset(base_matcher, dataset, config, device, method_name=args.baseline_name)
        baseline_summary = aggregate_benchmark_results(
            base_raw, base_vram, config, method_name=args.baseline_name, ckpt_path=args.baseline_ckpt
        )
        summaries.append(baseline_summary)
        del base_matcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5. Primary Model Benchmark
    logger.info(f"--- Step 2/2: Benchmarking Target Model [{args.method_name}] ---")
    target_matcher = load_benchmark_model(args.ckpt, config, device, method_name=args.method_name)
    target_raw, target_vram = evaluate_model_on_dataset(target_matcher, dataset, config, device, method_name=args.method_name)
    target_summary = aggregate_benchmark_results(
        target_raw, target_vram, config, method_name=args.method_name, ckpt_path=args.ckpt
    )
    summaries.append(target_summary)

    # 6. Save Artifacts, CSV, Reports, Plots
    save_benchmark_artifacts(target_summary, out_dir, baseline_summary=baseline_summary)
    if baseline_summary:
        # Also save baseline summary into baseline subfolder or alongside
        base_dir = out_dir / "baseline"
        save_benchmark_artifacts(baseline_summary, base_dir)

    # 7. Print Master Terminal Comparison Tables
    print_terminal_tables(summaries)

    print(f"\n[✓] Benchmark complete! All artifacts and plots saved to:\n    {out_dir}\n")


if __name__ == '__main__':
    main()
