#!/usr/bin/env python3
"""
benchmark_all_models.py - Unified Multi-Model Benchmark Suite on StereoLunar

Evaluates:
  1. Fine-tuned EfficientLoFTR (lunar specialized)
  2. Base EfficientLoFTR (pretrained outdoor foundation)
  3. SuperPoint + LightGlue (vismatch learned keypoints + transformer matcher)
  4. SIFT + LightGlue (vismatch classical detector + transformer matcher)
  5. ALIKED + LightGlue (vismatch deformable learned keypoints + transformer matcher)
  6. LoFTR (vismatch dense detector-free foundation baseline)

Evaluates on the held-out StereoLunar test set:
  - Direct 3D Correspondence: MMA@1px, 2px, 3px, 5px, 10px, RMSE (px), Precision@3px, Precision@5px.
  - 6-DoF Stereo Pose (baseline >= 1.0m): AUC@5°, 10°, 20°, Pose Success @5°, 10°, 20°, R_err, t_err.
  - 3-DoF Pure Rotation Zero-Baseline (baseline < 1.0m): AUC@5°, 10°, 20°, Rot Success @1°, 3°, 5°, 10°, R_err.
  - Efficiency: Avg matches/pair, latency (ms/pair), throughput (pairs/sec), peak VRAM.
  - Stratified Illumination Breakdown (Same ΔSun=0°, Moderate ΔSun, Severe ΔSun).
  - Rich Visual Registration Artifacts: Match lines, checkerboard mosaics, error heatmaps, multi-model grids.
  - Leaderboard CSV, summary JSON, plots, and a publication-ready Markdown report.
"""

import os
import sys
import time
import json
import csv
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "EfficientLoFTR"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import cv2
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
import vismatch


# ==============================================================================
# 1. Zero Data Leakage Verification
# ==============================================================================
def verify_dataset_integrity_and_leakage(split_file_path):
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

    if overlap_train_test or overlap_val_test or overlap_train_val:
        raise AssertionError("DATA LEAKAGE DETECTED! Scene overlap found across splits.")

    logger.info(
        f"[✓] Zero Data Leakage Verified: "
        f"{len(train_scene_ids)} Train Scenes ({np.sum(train_mask)} pairs), "
        f"{len(val_scene_ids)} Val Scenes ({np.sum(val_mask)} pairs), "
        f"{len(test_scene_ids)} Test Scenes ({np.sum(test_mask)} pairs)."
    )

    return {
        'train_scenes': len(train_scene_ids),
        'val_scenes': len(val_scene_ids),
        'test_scenes': len(test_scene_ids),
        'test_pairs': int(np.sum(test_mask))
    }


# ==============================================================================
# 2. Model Loading Helpers
# ==============================================================================
def load_eloftr_model(ckpt_path, device, name="EfficientLoFTR"):
    logger.info(f"Loading {name} from {ckpt_path}...")
    config = get_cfg_defaults()
    config.merge_from_file(str(PROJECT_ROOT / "configs" / "loftr" / "eloftr_lunar.py"))
    config.merge_from_file(str(PROJECT_ROOT / "configs" / "data" / "stereo_lunar_512.py"))
    _cfg = lower_config(config)

    matcher = LoFTR(config=_cfg['loftr'])
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        matcher.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found; initializing without checkpoint.")

    matcher = reparameter(matcher).eval().to(device)
    for p in matcher.parameters():
        p.requires_grad = False
    return matcher, config


def get_matcher_instance(model_key, device, custom_ckpt=None):
    """
    Instantiates the requested model (returns (matcher, model_type, config)).
    model_type: 'eloftr' or 'vismatch'
    """
    if model_key == 'eloftr_finetuned':
        ckpt = custom_ckpt or str(PROJECT_ROOT / "logs/tb_logs/lunar_full_finetune/version_1/checkpoints/epoch=0-auc@5=0.788-auc@10=0.853-auc@20=0.899.ckpt")
        matcher, cfg = load_eloftr_model(ckpt, device, name="Fine-Tuned EfficientLoFTR")
        return matcher, 'eloftr', cfg
    elif model_key == 'eloftr_base':
        ckpt = custom_ckpt or str(PROJECT_ROOT / "weights/eloftr_outdoor.ckpt")
        matcher, cfg = load_eloftr_model(ckpt, device, name="Base EfficientLoFTR (Outdoor)")
        return matcher, 'eloftr', cfg
    elif model_key in ['superpoint-lightglue', 'sift-lightglue', 'aliked-lightglue', 'loftr']:
        logger.info(f"Loading [{model_key}] via vismatch on {device}...")
        m = vismatch.get_matcher(model_key, device=device)
        cfg = get_cfg_defaults()
        cfg.merge_from_file(str(PROJECT_ROOT / "configs" / "loftr" / "eloftr_lunar.py"))
        cfg.merge_from_file(str(PROJECT_ROOT / "configs" / "data" / "stereo_lunar_512.py"))
        return m, 'vismatch', cfg
    else:
        raise ValueError(f"Unsupported model key: {model_key}")


# ==============================================================================
# 3. Geometric & Correspondence Evaluation
# ==============================================================================
def compute_correspondence_metrics(mkpts0, mkpts1, depth0, depth1, T_0to1, K0, K1):
    M = len(mkpts0)
    if M == 0:
        return {
            'mma@1': 0.0, 'mma@2': 0.0, 'mma@3': 0.0, 'mma@5': 0.0, 'mma@10': 0.0,
            'rmse_px': np.nan, 'precision@3': 0.0, 'precision@5': 0.0,
            'valid_gt_count': 0, 'corr_errors': np.array([])
        }

    kpts0 = mkpts0.unsqueeze(0)
    valid_mask, w_kpts0 = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1)
    vmask = valid_mask[0]
    valid_count = int(vmask.sum().item())

    if valid_count == 0:
        return {
            'mma@1': 0.0, 'mma@2': 0.0, 'mma@3': 0.0, 'mma@5': 0.0, 'mma@10': 0.0,
            'rmse_px': np.nan, 'precision@3': 0.0, 'precision@5': 0.0,
            'valid_gt_count': 0, 'corr_errors': np.array([])
        }

    pred_pts1 = mkpts1[vmask]
    true_pts1 = w_kpts0[0][vmask]
    corr_errs = torch.norm(pred_pts1 - true_pts1, dim=-1).cpu().numpy()

    return {
        'mma@1': float(np.mean(corr_errs <= 1.0)) * 100.0,
        'mma@2': float(np.mean(corr_errs <= 2.0)) * 100.0,
        'mma@3': float(np.mean(corr_errs <= 3.0)) * 100.0,
        'mma@5': float(np.mean(corr_errs <= 5.0)) * 100.0,
        'mma@10': float(np.mean(corr_errs <= 10.0)) * 100.0,
        'rmse_px': float(np.sqrt(np.mean(corr_errs ** 2))),
        'precision@3': float((corr_errs <= 3.0).sum() / M) * 100.0,
        'precision@5': float((corr_errs <= 5.0).sum() / M) * 100.0,
        'valid_gt_count': valid_count,
        'corr_errors': corr_errs
    }


def match_single_pair(matcher, model_type, sample, device):
    """
    Runs forward matching on a single dataset sample dict.
    Returns: mkpts0 (Tensor [N, 2]), mkpts1 (Tensor [N, 2]), latency_ms (float)
    """
    if model_type == 'eloftr':
        batch = {
            'image0': sample['image0'].unsqueeze(0).to(device),
            'image1': sample['image1'].unsqueeze(0).to(device)
        }
        t0 = time.perf_counter()
        with torch.inference_mode():
            matcher(batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return batch['mkpts0_f'], batch['mkpts1_f'], (t1 - t0) * 1000.0
    else:  # vismatch
        img0_3ch = sample['image0'].repeat(3, 1, 1).to(device)
        img1_3ch = sample['image1'].repeat(3, 1, 1).to(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = matcher(img0_3ch, img1_3ch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        mk0 = torch.from_numpy(out['matched_kpts0']).float().to(device)
        mk1 = torch.from_numpy(out['matched_kpts1']).float().to(device)
        return mk0, mk1, (t1 - t0) * 1000.0


# ==============================================================================
# 4. Model Evaluation Loop
# ==============================================================================
def evaluate_model(model_key, dataset, device, custom_ckpt=None):
    logger.info(f"\n=======================================================")
    logger.info(f"BENCHMARKING MODEL: {model_key}")
    logger.info(f"=======================================================")

    matcher, model_type, config = get_matcher_instance(model_key, device, custom_ckpt)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    records = []
    runtimes = []
    zero_baseline_thr = getattr(config.TRAINER, 'ZERO_BASELINE_THR', 1.0)

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating {model_key}"):
        sample = dataset[idx]

        mkpts0, mkpts1, lat_ms = match_single_pair(matcher, model_type, sample, device)
        runtimes.append(lat_ms)

        depth0 = sample['depth0'].unsqueeze(0).to(device)
        depth1 = sample['depth1'].unsqueeze(0).to(device)
        T_0to1 = sample['T_0to1'].unsqueeze(0).to(device)
        K0 = sample['K0'].unsqueeze(0).to(device)
        K1 = sample['K1'].unsqueeze(0).to(device)

        corr_res = compute_correspondence_metrics(mkpts0, mkpts1, depth0, depth1, T_0to1, K0, K1)

        eval_data = {
            'mkpts0_f': mkpts0,
            'mkpts1_f': mkpts1,
            'm_bids': torch.zeros(len(mkpts0), dtype=torch.long, device=device),
            'K0': K0,
            'K1': K1,
            'T_0to1': T_0to1
        }
        compute_symmetrical_epipolar_errors(eval_data)
        compute_pose_errors(eval_data, config)

        r_err = float(eval_data['R_errs'][0][0])
        t_err = float(eval_data['t_errs'][0][0])
        pose_err = max(r_err, t_err)
        epi_err = float(eval_data['epi_errs'].mean().item()) if len(eval_data['epi_errs']) > 0 else np.nan

        rel_names = sample['pair_names']
        pair_id = f"{rel_names[0]}#{rel_names[1]}"
        scene = rel_names[0].split('/')[0] if '/' in rel_names[0] else str(sample.get('scene_id', 'unknown'))
        baseline = float(torch.norm(T_0to1[0, :3, 3]).item())
        is_stereo = (baseline >= zero_baseline_thr)

        records.append({
            'idx': idx,
            'pair_id': pair_id,
            'scene': scene,
            'score': float(sample.get('score', 0.0)),
            'baseline_m': baseline,
            'is_stereo': is_stereo,
            'num_matches': len(mkpts0),
            'valid_gt_count': corr_res['valid_gt_count'],
            'mma@1': corr_res['mma@1'],
            'mma@2': corr_res['mma@2'],
            'mma@3': corr_res['mma@3'],
            'mma@5': corr_res['mma@5'],
            'mma@10': corr_res['mma@10'],
            'rmse_px': corr_res['rmse_px'],
            'precision@3': corr_res['precision@3'],
            'precision@5': corr_res['precision@5'],
            'R_err_deg': r_err,
            't_err_deg': t_err,
            'pose_err_deg': pose_err,
            'epi_err_px2': epi_err,
            'runtime_ms': lat_ms
        })

    # Cleanup model from VRAM
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == 'cuda' else 0.0
    del matcher
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    summary = aggregate_model_metrics(records, runtimes, peak_vram_gb, model_key)
    return summary, records


def aggregate_model_metrics(records, runtimes, peak_vram_gb, model_key):
    n_pairs = len(records)
    stereo_records = [r for r in records if r['is_stereo']]
    zerob_records = [r for r in records if not r['is_stereo']]

    # 1. 3D Correspondence
    all_mma1 = np.array([r['mma@1'] for r in records])
    all_mma2 = np.array([r['mma@2'] for r in records])
    all_mma3 = np.array([r['mma@3'] for r in records])
    all_mma5 = np.array([r['mma@5'] for r in records])
    all_mma10 = np.array([r['mma@10'] for r in records])
    valid_rmses = [r['rmse_px'] for r in records if not np.isnan(r['rmse_px'])]
    all_prec3 = np.array([r['precision@3'] for r in records])
    all_prec5 = np.array([r['precision@5'] for r in records])
    all_matches = np.array([r['num_matches'] for r in records])

    # 2. 6-DoF Stereo Pose
    stereo_pose_errs = np.array([r['pose_err_deg'] for r in stereo_records]) if stereo_records else np.array([np.inf])
    stereo_r_errs = np.array([r['R_err_deg'] for r in stereo_records]) if stereo_records else np.array([np.inf])
    stereo_t_errs = np.array([r['t_err_deg'] for r in stereo_records]) if stereo_records else np.array([np.inf])

    s_aucs = error_auc(stereo_pose_errs, [5, 10, 20])
    stereo_auc5 = float(s_aucs['auc@5']) * 100.0
    stereo_auc10 = float(s_aucs['auc@10']) * 100.0
    stereo_auc20 = float(s_aucs['auc@20']) * 100.0

    pose_succ_5 = float(np.mean(stereo_pose_errs < 5.0)) * 100.0 if len(stereo_records) else 0.0
    pose_succ_10 = float(np.mean(stereo_pose_errs < 10.0)) * 100.0 if len(stereo_records) else 0.0
    pose_succ_20 = float(np.mean(stereo_pose_errs < 20.0)) * 100.0 if len(stereo_records) else 0.0

    # 3. 3-DoF Zero-Baseline Rotation
    zerob_r_errs = np.array([r['R_err_deg'] for r in zerob_records]) if zerob_records else np.array([np.inf])
    z_aucs = error_auc(zerob_r_errs, [5, 10, 20])
    zerob_auc5 = float(z_aucs['auc@5']) * 100.0
    zerob_auc10 = float(z_aucs['auc@10']) * 100.0
    zerob_auc20 = float(z_aucs['auc@20']) * 100.0

    rot_succ_1 = float(np.mean(zerob_r_errs < 1.0)) * 100.0 if len(zerob_records) else 0.0
    rot_succ_3 = float(np.mean(zerob_r_errs < 3.0)) * 100.0 if len(zerob_records) else 0.0
    rot_succ_5 = float(np.mean(zerob_r_errs < 5.0)) * 100.0 if len(zerob_records) else 0.0

    scores = np.array([r['score'] for r in records])
    severe_mask = (scores >= 0.6)
    moderate_mask = (scores >= 0.2) & (scores < 0.6)
    same_mask = (scores < 0.2)

    return {
        'model': model_key,
        'num_pairs': n_pairs,
        'num_stereo': len(stereo_records),
        'num_zerob': len(zerob_records),
        'avg_matches': float(np.mean(all_matches)),
        'median_matches': float(np.median(all_matches)),
        'avg_latency_ms': float(np.mean(runtimes)),
        'throughput_pairs_sec': float(1000.0 / np.mean(runtimes)) if np.mean(runtimes) > 0 else 0.0,
        'peak_vram_gb': float(peak_vram_gb),
        'mma@1': float(np.mean(all_mma1)),
        'mma@2': float(np.mean(all_mma2)),
        'mma@3': float(np.mean(all_mma3)),
        'mma@5': float(np.mean(all_mma5)),
        'mma@10': float(np.mean(all_mma10)),
        'rmse_px': float(np.mean(valid_rmses)) if valid_rmses else np.nan,
        'precision@3': float(np.mean(all_prec3)),
        'precision@5': float(np.mean(all_prec5)),
        'stereo_auc@5': stereo_auc5,
        'stereo_auc@10': stereo_auc10,
        'stereo_auc@20': stereo_auc20,
        'stereo_pose_succ@5': pose_succ_5,
        'stereo_pose_succ@10': pose_succ_10,
        'stereo_pose_succ@20': pose_succ_20,
        'stereo_median_r_err': float(np.median(stereo_r_errs)),
        'stereo_median_t_err': float(np.median(stereo_t_errs)),
        'zerob_auc@5': zerob_auc5,
        'zerob_auc@10': zerob_auc10,
        'zerob_auc@20': zerob_auc20,
        'zerob_rot_succ@1': rot_succ_1,
        'zerob_rot_succ@3': rot_succ_3,
        'zerob_rot_succ@5': rot_succ_5,
        'zerob_median_r_err': float(np.median(zerob_r_errs)),
        'mma@5_same_sun': float(np.mean(all_mma5[same_mask])) if np.sum(same_mask) > 0 else np.nan,
        'mma@5_moderate_delta': float(np.mean(all_mma5[moderate_mask])) if np.sum(moderate_mask) > 0 else np.nan,
        'mma@5_severe_delta': float(np.mean(all_mma5[severe_mask])) if np.sum(severe_mask) > 0 else np.nan,
    }


# ==============================================================================
# 5. Visual Registration Artifacts
# ==============================================================================
def create_checkerboard(img1, img2, tile_size=64):
    h, w = img1.shape[:2]
    checker = img1.copy()
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if ((y // tile_size) + (x // tile_size)) % 2 == 1:
                checker[y:min(y+tile_size, h), x:min(x+tile_size, w)] = \
                    img2[y:min(y+tile_size, h), x:min(x+tile_size, w)]
    return checker


def draw_matches_panel(img0_u8, img1_u8, kpts0, kpts1, max_draw=120):
    h0, w0 = img0_u8.shape[:2]
    h1, w1 = img1_u8.shape[:2]
    gap = 10
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1 + gap

    img0_rgb = cv2.cvtColor(img0_u8, cv2.COLOR_GRAY2RGB) if len(img0_u8.shape) == 2 else img0_u8.copy()
    img1_rgb = cv2.cvtColor(img1_u8, cv2.COLOR_GRAY2RGB) if len(img1_u8.shape) == 2 else img1_u8.copy()

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30
    canvas[:h0, :w0] = img0_rgb
    canvas[:h1, w0+gap:] = img1_rgb

    n_matches = len(kpts0)
    if n_matches > 0:
        step = max(1, n_matches // max_draw)
        sub_indices = list(range(0, n_matches, step))[:max_draw]
        cmap = plt.get_cmap('spring')
        for i, idx in enumerate(sub_indices):
            pt0 = (int(round(kpts0[idx][0])), int(round(kpts0[idx][1])))
            pt1 = (int(round(kpts1[idx][0])) + w0 + gap, int(round(kpts1[idx][1])))
            c = cmap(i / max(1, len(sub_indices)))[:3]
            color = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt0, 3, (0, 255, 0), -1)
            cv2.circle(canvas, pt1, 3, (0, 255, 0), -1)

    return canvas


def generate_visual_artifacts_for_pair(pair_idx, dataset, model_keys, out_dir, device, custom_ckpts=None):
    logger.info(f"Generating visual registration artifacts for Showcase Pair {pair_idx}...")
    sample = dataset[pair_idx]
    img0_np = (sample['image0'].numpy()[0] * 255).astype(np.uint8)
    img1_np = (sample['image1'].numpy()[0] * 255).astype(np.uint8)

    pair_out_dir = out_dir / "registration_visualizations" / f"pair_{pair_idx}"
    pair_out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(pair_out_dir / "img0_reference.png"), img0_np)
    cv2.imwrite(str(pair_out_dir / "img1_query.png"), img1_np)

    results = {}
    for m_key in model_keys:
        matcher, m_type, cfg = get_matcher_instance(m_key, device, custom_ckpts.get(m_key) if custom_ckpts else None)
        mkpts0, mkpts1, _ = match_single_pair(matcher, m_type, sample, device)
        mk0_np = mkpts0.cpu().numpy()
        mk1_np = mkpts1.cpu().numpy()

        match_vis = draw_matches_panel(img0_np, img1_np, mk0_np, mk1_np)
        cv2.imwrite(str(pair_out_dir / f"{m_key}_matches.png"), cv2.cvtColor(match_vis, cv2.COLOR_RGB2BGR))

        H = None
        warped_img0 = np.zeros_like(img1_np)
        checkerboard = img1_np.copy()
        diff_heatmap = np.zeros_like(img1_np)

        if len(mk0_np) >= 4:
            H, mask = cv2.findHomography(mk0_np, mk1_np, cv2.RANSAC, 3.0)
            if H is not None:
                warped_img0 = cv2.warpPerspective(img0_np, H, (img1_np.shape[1], img1_np.shape[0]))
                checkerboard = create_checkerboard(img1_np, warped_img0, tile_size=48)
                diff = cv2.absdiff(img1_np, warped_img0)
                diff_heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_VIRIDIS)
                valid_warp = (warped_img0 > 0).astype(np.uint8)
                diff_heatmap[valid_warp == 0] = 0

        cv2.imwrite(str(pair_out_dir / f"{m_key}_checkerboard.png"), checkerboard)
        cv2.imwrite(str(pair_out_dir / f"{m_key}_diff_heatmap.png"), diff_heatmap)

        results[m_key] = {
            'num_matches': len(mk0_np),
            'match_vis': match_vis,
            'checkerboard': checkerboard,
            'diff_heatmap': diff_heatmap,
            'H': H
        }

        del matcher
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    n_models = len(model_keys)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 4.5 * n_models), dpi=150)
    if n_models == 1:
        axes = np.array([axes])

    for i, m_key in enumerate(model_keys):
        axes[i, 0].imshow(results[m_key]['match_vis'])
        axes[i, 0].set_title(f"{m_key.upper()} — {results[m_key]['num_matches']} Matches", fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(results[m_key]['checkerboard'], cmap='gray')
        axes[i, 1].set_title(f"{m_key.upper()} — Checkerboard Registration", fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

    plt.suptitle(f"Multi-Model Visual Registration Showcase: Test Pair #{pair_idx}", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    grid_path = pair_out_dir / f"pair_{pair_idx}_multi_model_comparison.png"
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Showcase grid saved to: {grid_path}")


# ==============================================================================
# 6. Comparison Plots & Reports Generation
# ==============================================================================
def generate_comparison_plots(summaries, out_dir):
    logger.info("Generating comparative benchmark plots...")
    models = [s['model'] for s in summaries]
    clean_labels = [m.replace('_', ' ').replace('-', '+\n') for m in models]
    x = np.arange(len(models))
    colors = ['#2ca02c' if 'finetuned' in m else '#1f77b4' if 'eloftr' in m else '#ff7f0e' for m in models]

    # Plot 1: 3D Surface Correspondence Accuracy (MMA@1, 3, 5, 10px)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    w = 0.18
    ax.bar(x - 1.5*w, [s['mma@1'] for s in summaries], w, label='MMA@1px', color='#6baed6')
    ax.bar(x - 0.5*w, [s['mma@3'] for s in summaries], w, label='MMA@3px', color='#3182bd')
    ax.bar(x + 0.5*w, [s['mma@5'] for s in summaries], w, label='MMA@5px', color='#08519c')
    ax.bar(x + 1.5*w, [s['mma@10'] for s in summaries], w, label='MMA@10px', color='#08306b')
    ax.set_ylabel('Mean Matching Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('3D Surface Correspondence Accuracy on StereoLunar (MMA @ Thresholds)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_mma_comparison.png")
    plt.close()

    # Plot 2: 6-DoF & 3-DoF Pose AUC Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    bars1 = ax1.bar(x, [s['stereo_auc@10'] for s in summaries], color=colors, width=0.55, edgecolor='black')
    ax1.set_ylabel('AUC@10° (%)', fontsize=11, fontweight='bold')
    ax1.set_title('6-DoF Genuine Stereo Pose (AUC@10°)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_labels, fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    bars2 = ax2.bar(x, [s['zerob_auc@10'] for s in summaries], color=colors, width=0.55, edgecolor='black')
    ax2.set_ylabel('AUC@10° (%)', fontsize=11, fontweight='bold')
    ax2.set_title('3-DoF Zero-Baseline Rotation (AUC@10°)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(clean_labels, fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Pose Recovery Under Complex Lunar Conditions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_pose_auc_comparison.png")
    plt.close()

    # Plot 3: Latency vs Accuracy Pareto Curve
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for s in summaries:
        m = s['model']
        c = '#2ca02c' if 'finetuned' in m else '#1f77b4' if 'eloftr' in m else '#ff7f0e'
        ax.scatter(s['avg_latency_ms'], s['mma@5'], s=s['avg_matches']/5 + 80, color=c, edgecolors='black', alpha=0.85)
        ax.annotate(m.replace('_', ' '), (s['avg_latency_ms'], s['mma@5']),
                    xytext=(8, 4), textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_xlabel('Inference Latency per Pair (ms) [Lower is Better]', fontsize=11, fontweight='bold')
    ax.set_ylabel('MMA@5px (%) [Higher is Better]', fontsize=11, fontweight='bold')
    ax.set_title('Efficiency vs Accuracy Pareto Frontier (Bubble Size = Avg Matches)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_latency_vs_accuracy.png")
    plt.close()

    # Plot 4: Illumination Delta Robustness Breakdown
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    w = 0.25
    ax.bar(x - w, [s.get('mma@5_same_sun', 0.0) for s in summaries], w, label='Same Illumination (ΔSun=0°)', color='#2ca02c')
    ax.bar(x, [s.get('mma@5_moderate_delta', 0.0) for s in summaries], w, label='Moderate Illumination Shift', color='#ff7f0e')
    ax.bar(x + w, [s.get('mma@5_severe_delta', 0.0) for s in summaries], w, label='Severe Illumination Shift', color='#d62728')
    ax.set_ylabel('MMA@5px (%)', fontsize=11, fontweight='bold')
    ax.set_title('Illumination Robustness Breakdown (MMA@5px)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_illumination_breakdown.png")
    plt.close()


def generate_markdown_report(summaries, leakage_info, out_dir, vis_pairs):
    report_path = out_dir / "benchmark_report.md"
    logger.info(f"Generating publication-ready Markdown report: {report_path}")

    md = []
    md.append("# Comprehensive Multi-Model Benchmark Report: StereoLunar Test Split\n")
    md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"**Test Set Guarantee:** Zero Data Leakage Verified (`Scenes(train) ∩ Scenes(test) = ∅`)  ")
    md.append(f"**Disjoint Test Partition:** {leakage_info['test_scenes']} scenes, {summaries[0]['num_pairs']} pairs evaluated ({summaries[0]['num_stereo']} 6-DoF Stereo, {summaries[0]['num_zerob']} 3-DoF Pure Rotation)  \n")

    md.append("## 1. Executive Summary Leaderboard\n")
    md.append("| Model | MMA@3px (%) | MMA@5px (%) | Precision@5 (%) | 6-DoF Stereo AUC@10° | 3-DoF Rot AUC@10° | Avg Matches | Latency (ms) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in summaries:
        name_fmt = f"**{s['model']}**" if 'finetuned' in s['model'] else s['model']
        md.append(
            f"| {name_fmt} | {s['mma@3']:.2f}% | {s['mma@5']:.2f}% | {s['precision@5']:.2f}% | "
            f"{s['stereo_auc@10']:.2f}% | {s['zerob_auc@10']:.2f}% | {s['avg_matches']:.1f} | {s['avg_latency_ms']:.1f} ms |"
        )
    md.append("\n")

    md.append("## 2. Granular 3D Ground-Truth Surface Correspondence Metrics\n")
    md.append("Evaluated using camera intrinsics $K$ and ground-truth sub-millimeter unprojected surface depth maps via `warp_kpts`:\n")
    md.append("| Model | MMA@1px | MMA@2px | MMA@3px | MMA@5px | MMA@10px | RMSE (px) | Precision@3px | Precision@5px |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in summaries:
        md.append(
            f"| {s['model']} | {s['mma@1']:.2f}% | {s['mma@2']:.2f}% | {s['mma@3']:.2f}% | "
            f"{s['mma@5']:.2f}% | {s['mma@10']:.2f}% | {s['rmse_px']:.2f} px | {s['precision@3']:.2f}% | {s['precision@5']:.2f}% |"
        )
    md.append("\n")

    md.append("## 3. 6-DoF Genuine Stereo Pose Estimation (Baseline $\\ge$ 1.0m)\n")
    md.append("| Model | AUC@5° | AUC@10° | AUC@20° | Success@5° | Success@10° | Median R Err | Median t Err |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in summaries:
        md.append(
            f"| {s['model']} | {s['stereo_auc@5']:.2f}% | {s['stereo_auc@10']:.2f}% | {s['stereo_auc@20']:.2f}% | "
            f"{s['stereo_pose_succ@5']:.2f}% | {s['stereo_pose_succ@10']:.2f}% | {s['stereo_median_r_err']:.2f}° | {s['stereo_median_t_err']:.2f}° |"
        )
    md.append("\n")

    md.append("## 4. 3-DoF Pure Rotation Zero-Baseline Registration (Illumination Drift)\n")
    md.append("| Model | AUC@5° | AUC@10° | AUC@20° | Success@1° | Success@3° | Success@5° | Median R Err |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in summaries:
        md.append(
            f"| {s['model']} | {s['zerob_auc@5']:.2f}% | {s['zerob_auc@10']:.2f}% | {s['zerob_auc@20']:.2f}% | "
            f"{s['zerob_rot_succ@1']:.2f}% | {s['zerob_rot_succ@3']:.2f}% | {s['zerob_rot_succ@5']:.2f}% | {s['zerob_median_r_err']:.2f}° |"
        )
    md.append("\n")

    md.append("## 5. Illumination Robustness Stratification (MMA@5px)\n")
    md.append("| Model | Same Sun Angle (ΔSun=0°) | Moderate Shift | Severe Shift | Delta Drop |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for s in summaries:
        drop = s.get('mma@5_same_sun', 0.0) - s.get('mma@5_severe_delta', 0.0)
        md.append(
            f"| {s['model']} | {s.get('mma@5_same_sun', 0.0):.2f}% | {s.get('mma@5_moderate_delta', 0.0):.2f}% | "
            f"{s.get('mma@5_severe_delta', 0.0):.2f}% | {drop:+.2f}% |"
        )
    md.append("\n")

    md.append("## 6. Runtime & Computational Efficiency\n")
    md.append("| Model | Latency (ms/pair) | Throughput (pairs/s) | Peak VRAM (GB) | Avg Matches |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for s in summaries:
        md.append(
            f"| {s['model']} | {s['avg_latency_ms']:.2f} ms | {s['throughput_pairs_sec']:.1f} | "
            f"{s['peak_vram_gb']:.2f} GB | {s['avg_matches']:.1f} |"
        )
    md.append("\n")

    md.append("## 7. Comparative Benchmark Visualizations\n")
    md.append("- **3D Surface Correspondence:** `benchmark_mma_comparison.png`\n")
    md.append("- **Pose Recovery AUC:** `benchmark_pose_auc_comparison.png`\n")
    md.append("- **Pareto Efficiency vs Accuracy:** `benchmark_latency_vs_accuracy.png`\n")
    md.append("- **Illumination Robustness Breakdown:** `benchmark_illumination_breakdown.png`\n")

    if vis_pairs:
        md.append("### Showcase Registration Examples\n")
        for p_idx in vis_pairs:
            md.append(f"- **Pair #{p_idx} Multi-Model Comparison:** `registration_visualizations/pair_{p_idx}/pair_{p_idx}_multi_model_comparison.png`\n")

    with open(report_path, 'w') as f:
        f.write("\n".join(md))
    logger.info(f"[✓] Benchmark report written to {report_path}")


# ==============================================================================
# 7. Main CLI Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Model Benchmark Suite on StereoLunar")
    parser.add_argument('--models', type=str,
                        default='eloftr_finetuned,eloftr_base,superpoint-lightglue,sift-lightglue,aliked-lightglue,loftr',
                        help='Comma-separated list of models to benchmark')
    parser.add_argument('--stride', type=int, default=10,
                        help='Sampling stride over the 9,186 test pairs (default: 10 => ~919 pairs; 1 => all 9,186 pairs)')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Max test pairs to evaluate (optional cap for ultra-fast runs)')
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT.parent / "benchmark_results" / "all_models_benchmark"),
                        help='Directory to store benchmark results and artifacts')
    parser.add_argument('--visualize_pairs', type=str, default='4418,744',
                        help='Comma-separated test pair indices for visual registration artifacts (default: 4418,744)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to evaluate on (cuda or cpu)')
    parser.add_argument('--finetuned_ckpt', type=str, default=None,
                        help='Path to fine-tuned EfficientLoFTR checkpoint (optional override)')
    parser.add_argument('--base_ckpt', type=str, default=None,
                        help='Path to base EfficientLoFTR checkpoint (optional override)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Verification & Loading
    split_file = PROJECT_ROOT / "data" / "stereo_lunar" / "split.npz"
    data_root = PROJECT_ROOT / "data" / "stereo_lunar" / "scenes"
    leakage_info = verify_dataset_integrity_and_leakage(str(split_file))

    dataset = StereoLunarDataset(
        root_dir=str(data_root),
        split_file=str(split_file),
        split='test',
        img_resize=512
    )

    indices = list(range(0, len(dataset), args.stride))
    if args.max_pairs is not None:
        indices = indices[:args.max_pairs]

    original_count = len(dataset)
    dataset.pairs = dataset.pairs[indices]
    logger.info(f"Subsampled test set from {original_count} to {len(dataset)} pairs (stride={args.stride}, max_pairs={args.max_pairs}).")

    model_list = [m.strip() for m in args.models.split(',') if m.strip()]
    vis_pairs = [int(p.strip()) for p in args.visualize_pairs.split(',') if p.strip()]

    custom_ckpts = {
        'eloftr_finetuned': args.finetuned_ckpt,
        'eloftr_base': args.base_ckpt
    }

    # 2. Evaluate Each Model
    all_summaries = []
    for model_key in model_list:
        summary, records = evaluate_model(model_key, dataset, device, custom_ckpts.get(model_key))
        all_summaries.append(summary)

        # Save per-model granular CSV
        model_csv = out_dir / f"per_pair_metrics_{model_key}.csv"
        with open(model_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        logger.info(f"Per-pair metrics for {model_key} saved to {model_csv}")

    # 3. Generate Visual Registration Artifacts for Showcase Pairs
    full_dataset = StereoLunarDataset(
        root_dir=str(data_root),
        split_file=str(split_file),
        split='test',
        img_resize=512
    )
    for p_idx in vis_pairs:
        if p_idx < len(full_dataset):
            generate_visual_artifacts_for_pair(p_idx, full_dataset, model_list, out_dir, device, custom_ckpts)

    # 4. Save Master Leaderboard & JSON
    leaderboard_csv = out_dir / "leaderboard_all_models.csv"
    with open(leaderboard_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)
    logger.info(f"[✓] Master leaderboard saved to: {leaderboard_csv}")

    summary_json = out_dir / "summary_all_models.json"
    with open(summary_json, 'w') as f:
        json.dump({
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'leakage_verified': leakage_info,
                'evaluated_pairs': len(dataset),
                'stride': args.stride,
                'models': model_list
            },
            'summaries': all_summaries
        }, f, indent=2)
    logger.info(f"[✓] Summary JSON saved to: {summary_json}")

    # 5. Generate Plots & Master Markdown Report
    generate_comparison_plots(all_summaries, out_dir)
    generate_markdown_report(all_summaries, leakage_info, out_dir, vis_pairs)

    # 6. Copy Key Artifacts to Antigravity CLI Brain
    brain_dir = Path("/home/ojas/.gemini/antigravity-cli/brain/85a283dc-b632-40a0-bc95-c3935e79b304")
    if brain_dir.exists():
        brain_out = brain_dir / "multi_model_benchmark"
        brain_out.mkdir(parents=True, exist_ok=True)
        for f_name in ["benchmark_report.md", "leaderboard_all_models.csv", "summary_all_models.json",
                       "benchmark_mma_comparison.png", "benchmark_pose_auc_comparison.png",
                       "benchmark_latency_vs_accuracy.png", "benchmark_illumination_breakdown.png"]:
            src_f = out_dir / f_name
            if src_f.exists():
                shutil.copy(src_f, brain_out / f_name)
        for p_idx in vis_pairs:
            grid_f = out_dir / "registration_visualizations" / f"pair_{p_idx}" / f"pair_{p_idx}_multi_model_comparison.png"
            if grid_f.exists():
                shutil.copy(grid_f, brain_out / f"pair_{p_idx}_multi_model_comparison.png")
        logger.info(f"[✓] Synced report and artifacts to brain directory: {brain_out}")

    logger.info("\n================================================================================")
    logger.info(f"COMPREHENSIVE BENCHMARK COMPLETED SUCCESSFULLY!")
    logger.info(f"Artifacts and Master Report Directory: {out_dir}")
    logger.info("================================================================================\n")


if __name__ == '__main__':
    main()
