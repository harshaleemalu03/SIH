"""Unified runner for lunar image registration benchmark.
Executes method-agnostic matching, shared RANSAC filtering, depth-warped GT reprojections,
masked photometric warp quality scoring, and standardized 10-metric aggregation over StereoLunar test split.
"""

import argparse
import gc
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Ensure finetune and EfficientLoFTR are on sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_FINETUNE_DIR = _CURRENT_DIR.parent
_REPO_ROOT = _FINETUNE_DIR.parent
_ELO_DIR = _FINETUNE_DIR / "EfficientLoFTR"

for p in [str(_REPO_ROOT), str(_FINETUNE_DIR), str(_ELO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from registration.adapters import (
    build_adapter,
    normalize_model_name,
    MatchOutput,
    ELOFTR_DEFAULT_CKPT,
    ELOFTR_BASE_CKPT,
)
from registration.gt import gt_reprojections, reprojection_errors
from registration.metrics import compute_pair_metrics, METRIC_COLUMNS
from registration.registration import warp_and_score

DEFAULT_MODELS = [
    "eloftr_finetuned",
    "eloftr_base",
    "roma",
    "superglue",
    "lightglue",
]

def verify_dataset_integrity_and_leakage(split_file_path: str | Path) -> Dict[str, int]:
    """Verify zero overlap between train, val, and test scenes."""
    p = Path(split_file_path)
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")

    data = np.load(str(p), allow_pickle=True)
    pairs = data["pairs"]
    split = data["split"]

    train_mask = (split == "train")
    val_mask = (split == "val")
    test_mask = (split == "test")

    train_scenes = set(pairs[train_mask][:, 0])
    val_scenes = set(pairs[val_mask][:, 0])
    test_scenes = set(pairs[test_mask][:, 0])

    overlap_train_test = train_scenes.intersection(test_scenes)
    overlap_val_test = val_scenes.intersection(test_scenes)
    overlap_train_val = train_scenes.intersection(val_scenes)

    if overlap_train_test or overlap_val_test or overlap_train_val:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED! Scene overlap: "
            f"train∩test={len(overlap_train_test)}, "
            f"val∩test={len(overlap_val_test)}, "
            f"train∩val={len(overlap_train_val)}"
        )

    return {
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "test_scenes": len(test_scenes),
        "test_pairs": int(np.sum(test_mask)),
    }

def create_checkerboard(img1: np.ndarray, img2: np.ndarray, tile_size: int = 64) -> np.ndarray:
    """Create checkerboard pattern between two images."""
    h, w = img1.shape[:2]
    checker = img1.copy()
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if ((y // tile_size) + (x // tile_size)) % 2 == 1:
                checker[y:min(y + tile_size, h), x:min(x + tile_size, w)] = \
                    img2[y:min(y + tile_size, h), x:min(x + tile_size, w)]
    return checker

def draw_matches_panel(
    img0_u8: np.ndarray,
    img1_u8: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    max_draw: int = 100,
) -> np.ndarray:
    """Draw side-by-side matches with inlier lines."""
    h0, w0 = img0_u8.shape[:2]
    h1, w1 = img1_u8.shape[:2]
    gap = 10
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1 + gap

    img0_bgr = cv2.cvtColor(img0_u8, cv2.COLOR_GRAY2BGR) if len(img0_u8.shape) == 2 else img0_u8.copy()
    img1_bgr = cv2.cvtColor(img1_u8, cv2.COLOR_GRAY2BGR) if len(img1_u8.shape) == 2 else img1_u8.copy()

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 35
    canvas[:h0, :w0] = img0_bgr
    canvas[:h1, w0 + gap:] = img1_bgr

    n = len(kpts0)
    if n > 0:
        step = max(1, n // max_draw)
        sub_indices = list(range(0, n, step))[:max_draw]
        for i, idx in enumerate(sub_indices):
            pt0 = (int(round(kpts0[idx][0])), int(round(kpts0[idx][1])))
            pt1 = (int(round(kpts1[idx][0])) + w0 + gap, int(round(kpts1[idx][1])))
            color = (0, 255, 0)
            cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt0, 3, (0, 255, 255), -1)
            cv2.circle(canvas, pt1, 3, (0, 255, 255), -1)

    return canvas

def evaluate_pair(adapter: Any, kind: str, sample: Dict[str, Any], device: str = "cuda") -> Dict[str, Any]:
    """Full 10-metric path for one dataset sample. Returns row dict."""
    img0 = sample["image0"].numpy()[0]
    img1 = sample["image1"].numpy()[0]

    if kind == "gray":
        out = adapter.match_gray(img0[None], img1[None])
    else:
        rgb0 = np.stack([img0, img0, img0], 0).astype(np.float32)
        rgb1 = np.stack([img1, img1, img1], 0).astype(np.float32)
        out = adapter.match_rgb(rgb0, rgb1)

    if len(out.inlier_kpts0) >= 1:
        valid, gt1 = gt_reprojections(
            out.inlier_kpts0,
            sample["depth0"],
            sample["depth1"],
            sample["T_0to1"],
            sample["K0"],
            sample["K1"],
        )
        errs = reprojection_errors(out.inlier_kpts1, gt1, valid)
    else:
        errs = np.array([])

    src = (np.stack([img0, img0, img0], -1) * 255).astype(np.uint8)
    ref = (np.stack([img1, img1, img1], -1) * 255).astype(np.uint8)
    src_bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    ref_bgr = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    quality = warp_and_score(src_bgr, ref_bgr, out.H)
    t_warp = (time.perf_counter() - t0) * 1000.0

    runtime = out.latency_match_ms + out.latency_ransac_ms + t_warp
    m = compute_pair_metrics(out, errs, (512, 512), runtime, quality)
    T = np.asarray(sample["T_0to1"]).reshape(4, 4)
    baseline_m = float(np.linalg.norm(T[:3, 3]))

    row = {
        "scene": str(sample.get("scene_id", "")),
        "pair_id": str(sample.get("pair_names", "")),
        "baseline_m": baseline_m,
        "score": float(sample.get("score", float("nan"))),
        "num_candidates": m.num_candidates,
        "num_inliers": m.num_inliers,
        "inlier_ratio": m.inlier_ratio,
        "reproj_rmse_px": m.reproj_rmse_px,
        "reproj_median_px": m.reproj_median_px,
        "reproj_max_px": m.reproj_max_px,
        "uniformity_score": m.uniformity_score,
        "spatial_coverage": m.spatial_coverage,
        "runtime_ms": m.runtime_ms,
        "reg_mse": m.registered_quality["mse"],
        "reg_psnr_db": m.registered_quality["psnr_db"],
        "reg_ssim": m.registered_quality["ssim"],
        "reg_ncc": m.registered_quality["ncc"],
        "overlap_ratio": m.registered_quality["overlap_ratio"],
        "_match_output": out,
        "_warped": quality.get("warped"),
        "_src_bgr": src_bgr,
        "_ref_bgr": ref_bgr,
    }
    return row

def run_benchmark(
    models: List[str],
    metrics: Optional[List[str]] = None,
    max_pairs: Optional[int] = None,
    stride: int = 1,
    device: str = "cuda",
    out_dir: str | Path = "finetune/benchmark_results/unified_registration",
    data_root: str | Path = "finetune/EfficientLoFTR/data/stereo_lunar/scenes",
    split_file: str | Path = "finetune/EfficientLoFTR/data/stereo_lunar/split.npz",
    custom_ckpt: Optional[str] = None,
    visuals: int = 3,
) -> pd.DataFrame:
    """Run benchmark across specified models and save results and reports."""
    from src.datasets.stereo_lunar import StereoLunarDataset
    from registration.report import summarize

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Zero data leakage verification
    leakage_info = verify_dataset_integrity_and_leakage(split_file)
    print(
        f"[✓] Zero Data Leakage Verified: "
        f"{leakage_info['train_scenes']} Train Scenes, "
        f"{leakage_info['val_scenes']} Val Scenes, "
        f"{leakage_info['test_scenes']} Test Scenes ({leakage_info['test_pairs']} test pairs)."
    )

    # 2. Instantiate held-out test split
    print(f"Loading StereoLunar test split (resize=512) from {data_root}...")
    dataset = StereoLunarDataset(
        root_dir=str(data_root),
        split_file=str(split_file),
        split="test",
        img_resize=512,
    )

    indices = list(range(0, len(dataset), max(1, stride)))
    if max_pairs is not None and max_pairs > 0:
        indices = indices[:max_pairs]
    print(f"Total test pairs to evaluate per model: {len(indices)} (stride={stride}, max={max_pairs})")

    all_rows = []
    device_obj = torch.device(device)

    for model_key in models:
        canonical_name = normalize_model_name(model_key)
        print(f"\n=======================================================")
        print(f"BENCHMARKING MODEL: {canonical_name} on {device}")
        print(f"=======================================================")

        try:
            adapter, kind = build_adapter(canonical_name, device=str(device_obj), custom_ckpt=custom_ckpt)
        except Exception as e:
            print(f"Error initializing model {canonical_name}: {e}")
            continue

        # Warmup on 5 samples (or fewer if dataset slice is small)
        warmup_count = min(5, len(dataset))
        print(f"Warming up {canonical_name} on {warmup_count} pairs (excluded from timing)...")
        for w_idx in range(warmup_count):
            try:
                evaluate_pair(adapter, kind, dataset[w_idx], device=str(device_obj))
            except Exception as e:
                print(f"Warmup warning for pair {w_idx}: {e}")

        model_rows = []
        visual_saved = 0
        vis_dir = out_path / "visualizations" / canonical_name
        if visuals > 0:
            vis_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(indices, desc=f"Evaluating {canonical_name}")
        for idx in pbar:
            sample = dataset[idx]
            try:
                row = evaluate_pair(adapter, kind, sample, device=str(device_obj))
                row["model"] = canonical_name

                # Save showcase visuals
                if visuals > 0 and visual_saved < visuals:
                    out_res = row.pop("_match_output")
                    warped = row.pop("_warped")
                    src_bgr = row.pop("_src_bgr")
                    ref_bgr = row.pop("_ref_bgr")

                    p_vis = vis_dir / f"pair_{idx}"
                    p_vis.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(p_vis / "ref_img0.png"), src_bgr)
                    cv2.imwrite(str(p_vis / "query_img1.png"), ref_bgr)

                    # Matches panel
                    matches_panel = draw_matches_panel(
                        src_bgr, ref_bgr, out_res.inlier_kpts0, out_res.inlier_kpts1
                    )
                    cv2.imwrite(str(p_vis / "inlier_matches.png"), matches_panel)

                    # Warped & Checkerboard
                    if warped is not None:
                        cv2.imwrite(str(p_vis / "warped_reg.png"), warped)
                        checker = create_checkerboard(ref_bgr, warped, tile_size=64)
                        cv2.imwrite(str(p_vis / "checkerboard_overlay.png"), checker)
                    visual_saved += 1
                else:
                    row.pop("_match_output", None)
                    row.pop("_warped", None)
                    row.pop("_src_bgr", None)
                    row.pop("_ref_bgr", None)

                model_rows.append(row)
                all_rows.append(row)

                # Update progress bar
                pbar.set_postfix(
                    inliers=row["num_inliers"],
                    rmse=f"{row['reproj_rmse_px']:.2f}" if np.isfinite(row["reproj_rmse_px"]) else "NaN",
                    ssim=f"{row['reg_ssim']:.2f}" if np.isfinite(row["reg_ssim"]) else "NaN",
                )
            except Exception as e:
                print(f"Error evaluating pair {idx} for {canonical_name}: {e}")

        # Save per-model CSV
        if model_rows:
            df_model = pd.DataFrame(model_rows)
            model_csv = out_path / f"per_pair_metrics_{canonical_name}.csv"
            df_model.to_csv(model_csv, index=False)
            print(f"Saved {len(model_rows)} rows to {model_csv}")

        # Explicitly release model weights and free GPU VRAM
        del adapter
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Combine all results
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        all_csv = out_path / "pairs_all.csv"
        df_all.to_csv(all_csv, index=False)
        print(f"\nSaved combined results ({len(df_all)} rows) to {all_csv}")

        # Generate summary leaderboard, plots, and markdown report
        print("Generating leaderboard, plots, and markdown report...")
        summarize(str(all_csv), str(out_path), selected_metrics=metrics)

    return pd.DataFrame(all_rows)

def parse_args():
    parser = argparse.ArgumentParser(description="Standardized Lunar Image Registration Benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated list of models to evaluate (default: {','.join(DEFAULT_MODELS)}). "
             f"Options: eloftr_finetuned, eloftr_base, roma, superglue, lightglue, loftr, sift-lightglue, etc.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Optional comma-separated list of metrics to filter / highlight in reports. "
             f"Available metrics: {','.join(METRIC_COLUMNS)}",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of test pairs to evaluate per model (default: all).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride over test dataset (default: 1). Use 10 or 50 for quick sweeps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="finetune/benchmark_results/unified_registration",
        help="Directory to save output CSVs, plots, and reports.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="finetune/EfficientLoFTR/data/stereo_lunar/scenes",
        help="Path to StereoLunar scenes folder.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="finetune/EfficientLoFTR/data/stereo_lunar/split.npz",
        help="Path to StereoLunar split.npz manifest.",
    )
    parser.add_argument(
        "--custom-ckpt",
        type=str,
        default=None,
        help="Custom checkpoint path for EfficientLoFTR.",
    )
    parser.add_argument(
        "--visuals",
        type=int,
        default=3,
        help="Number of visual showcase pairs to export per model (default: 3).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None

    run_benchmark(
        models=model_list,
        metrics=metric_list,
        max_pairs=args.max_pairs,
        stride=args.stride,
        device=args.device,
        out_dir=args.out,
        data_root=args.data_root,
        split_file=args.split_file,
        custom_ckpt=args.custom_ckpt,
        visuals=args.visuals,
    )

if __name__ == "__main__":
    main()
