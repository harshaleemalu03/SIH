"""Standardized Lunar Image Registration Benchmark CLI.
Provides a unified API and command-line interface for benchmarking image matchers
on the held-out StereoLunar test split with single shared RANSAC filtering and depth-warped GT.

Usage examples:
    python benchmark.py --models eloftr_finetuned,roma,superglue,lightglue
    python benchmark.py --models eloftr_finetuned,eloftr_base --metrics reproj_rmse_px,inlier_ratio,runtime_ms
    python benchmark.py --models roma --max-pairs 50 --stride 10 --device cuda
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

# Ensure finetune is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent
_FINETUNE_DIR = _REPO_ROOT / "finetune"
if str(_FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(_FINETUNE_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from registration.benchmark import run_benchmark, DEFAULT_MODELS
from registration.metrics import METRIC_COLUMNS

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

def display_summary_table(csv_path: Path, selected_metrics: Optional[List[str]] = None):
    """Render pretty summary table on terminal if rich is installed."""
    import pandas as pd
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if not _HAS_RICH:
        print("\n" + df.to_string())
        return

    console = Console()
    table = Table(title="[bold cyan]Standardized Lunar Image Registration Leaderboard[/bold cyan]", show_header=True)
    table.add_column("Model", style="bold green")

    # Select columns to display
    display_cols = []
    if selected_metrics:
        for c in df.columns:
            if c != "model" and any(m in c for m in selected_metrics):
                display_cols.append(c)
    if not display_cols:
        # Default core view
        core_prefixes = [
            "num_candidates_mean",
            "num_inliers_mean",
            "inlier_ratio_mean",
            "reproj_rmse_px_mean",
            "uniformity_score_mean",
            "spatial_coverage_mean",
            "runtime_ms_mean",
            "reg_ssim_mean",
            "reg_psnr_db_mean",
        ]
        display_cols = [c for c in core_prefixes if c in df.columns]

    for col in display_cols:
        table.add_column(col.replace("_mean", ""), justify="right")

    for _, row in df.iterrows():
        vals = [str(row["model"])]
        for col in display_cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}" if val == val else "NaN")
            else:
                vals.append(str(val))
        table.add_row(*vals)

    console.print()
    console.print(table)
    console.print()

def main():
    parser = argparse.ArgumentParser(
        description="Consistent, Standardized Lunar Image Registration Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated models to evaluate. Supported: eloftr_finetuned, eloftr_base, roma, superglue, lightglue, loftr. (Default: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=f"Comma-separated metrics to highlight or focus on. Available: {','.join(METRIC_COLUMNS)}",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of test pairs to evaluate per model.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Subsampling stride across the 9186-pair test split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="finetune/benchmark_results/unified_registration",
        help="Directory to write leaderboard, CSVs, plots, and markdown report.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="finetune/EfficientLoFTR/data/stereo_lunar/scenes",
        help="Path to StereoLunar scenes directory.",
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
        help="Custom checkpoint file for EfficientLoFTR.",
    )
    parser.add_argument(
        "--visuals",
        type=int,
        default=3,
        help="Number of visual showcase pairs to render per model.",
    )

    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None

    out_dir = Path(args.out)

    run_benchmark(
        models=model_list,
        metrics=metric_list,
        max_pairs=args.max_pairs,
        stride=args.stride,
        device=args.device,
        out_dir=out_dir,
        data_root=args.data_root,
        split_file=args.split_file,
        custom_ckpt=args.custom_ckpt,
        visuals=args.visuals,
    )

    leaderboard_csv = out_dir / "leaderboard.csv"
    if leaderboard_csv.exists():
        display_summary_table(leaderboard_csv, selected_metrics=metric_list)
        print(f"[✓] Benchmark finished successfully. Results saved in: {out_dir}")
        print(f"    - Leaderboard CSV: {leaderboard_csv}")
        print(f"    - Markdown Report: {out_dir / 'report.md'}")
        print(f"    - Analytical Plots: {out_dir / '*.png'}")

if __name__ == "__main__":
    main()
