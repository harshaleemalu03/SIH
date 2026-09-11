"""Leaderboard + plots + markdown generator for unified registration benchmark.
Separated from runner so reports can be regenerated headlessly from CSVs without GPUs.
"""

import json
from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from registration.metrics import METRIC_COLUMNS

AGG = {
    "num_candidates": "mean",
    "num_inliers": "mean",
    "inlier_ratio": "mean",
    "reproj_rmse_px": ["mean", "median"],
    "reproj_median_px": "median",
    "reproj_max_px": "median",
    "uniformity_score": "mean",
    "spatial_coverage": "mean",
    "runtime_ms": "mean",
    "reg_mse": "mean",
    "reg_psnr_db": "mean",
    "reg_ssim": "mean",
    "reg_ncc": "mean",
    "overlap_ratio": "mean",
}

def generate_plots(df: pd.DataFrame, out_dir: Path):
    """Generate the 4 analytical benchmark plots."""
    models = sorted(df["model"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(models))))

    # 1. RMSE CDF
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for i, m in enumerate(models):
        sub = df[df["model"] == m]
        v = np.sort(sub["reproj_rmse_px"].dropna().to_numpy())
        if len(v) > 0:
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=str(m), color=colors[i], lw=2)
    ax.set_xlabel("Reprojection RMSE per pair (px)")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical Cumulative Distribution of Reprojection RMSE")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.savefig(out_dir / "rmse_cdf.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Inlier Ratio vs Spatial Coverage
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for i, m in enumerate(models):
        sub = df[df["model"] == m]
        valid_mask = sub["spatial_coverage"].notna() & sub["inlier_ratio"].notna()
        sub_valid = sub[valid_mask]
        if len(sub_valid) > 0:
            cov = sub_valid["spatial_coverage"]
            ratio = sub_valid["inlier_ratio"]
            ax.scatter(cov, ratio, label=str(m), color=colors[i], alpha=0.5, edgecolors="none", s=30)
    ax.set_xlabel("Spatial Coverage (Fraction of 4x4 Cells Occupied)")
    ax.set_ylabel("Inlier Ratio")
    ax.set_title("Inlier Ratio vs Spatial Coverage")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.savefig(out_dir / "inlier_ratio_vs_coverage.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Runtime vs Reprojection RMSE
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for i, m in enumerate(models):
        sub = df[df["model"] == m]
        valid_mask = sub["runtime_ms"].notna() & sub["reproj_rmse_px"].notna()
        sub_valid = sub[valid_mask]
        if len(sub_valid) > 0:
            rt = sub_valid["runtime_ms"]
            rmse = sub_valid["reproj_rmse_px"]
            cands = sub_valid["num_candidates"].fillna(50)
            sizes = np.clip(cands / 10.0, 20, 200)
            ax.scatter(rt, rmse, label=str(m), color=colors[i], s=sizes, alpha=0.5)
    ax.set_xlabel("Total Runtime (ms)")
    ax.set_ylabel("Reprojection RMSE (px)")
    ax.set_title("Runtime vs Reprojection Error (Bubble Size ~ Candidate Matches)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.savefig(out_dir / "runtime_vs_rmse.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Illumination-Stratified RMSE
    if "score" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        bins = ["Same (<0.2)", "Moderate (0.2-0.6)", "Severe (>=0.6)"]
        x_indices = np.arange(len(bins))
        width = 0.8 / max(1, len(models))

        for i, m in enumerate(models):
            sub = df[df["model"] == m]
            same_rmse = sub[sub["score"] < 0.2]["reproj_rmse_px"].mean()
            mod_rmse = sub[(sub["score"] >= 0.2) & (sub["score"] < 0.6)]["reproj_rmse_px"].mean()
            sev_rmse = sub[sub["score"] >= 0.6]["reproj_rmse_px"].mean()
            vals = [
                same_rmse if np.isfinite(same_rmse) else 0.0,
                mod_rmse if np.isfinite(mod_rmse) else 0.0,
                sev_rmse if np.isfinite(sev_rmse) else 0.0,
            ]
            ax.bar(x_indices + i * width, vals, width, label=str(m), color=colors[i])

        ax.set_xlabel("Illumination Delta Category")
        ax.set_ylabel("Mean Reprojection RMSE (px)")
        ax.set_title("Illumination-Stratified Registration Accuracy")
        ax.set_xticks(x_indices + width * (len(models) - 1) / 2)
        ax.set_xticklabels(bins)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax.legend()
        fig.savefig(out_dir / "illumination_stratified.png", bbox_inches="tight")
        plt.close(fig)

def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(floatfmt=".3f")
    except Exception:
        lines = []
        headers = [str(df.index.name or "")] + [str(c) for c in df.columns]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for idx, row in df.iterrows():
            vals = [str(idx)]
            for val in row:
                if isinstance(val, (float, np.floating)):
                    vals.append(f"{val:.3f}" if np.isfinite(val) else "NaN")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

def build_markdown_report(
    leaderboard_df: pd.DataFrame,
    df: pd.DataFrame,
    out_dir: Path,
    selected_metrics: Optional[List[str]] = None,
) -> str:
    """Construct comprehensive markdown benchmark report."""
    md_lines = [
        "# Unified Lunar Image Registration Benchmark Report",
        "",
        "Evaluation performed on the held-out StereoLunar test set using single shared RANSAC filtering "
        "(`USAC_MAGSAC`, 3.0 px threshold, 2000 max iters) and depth-warped GT correspondences.",
        "",
        "## 1. Overall Leaderboard",
        "",
    ]

    # Convert leaderboard to markdown table
    md_lines.append(_df_to_markdown(leaderboard_df))
    md_lines.append("")

    if selected_metrics:
        md_lines.extend([
            "### Selected Metrics Focus",
            "",
            f"Filtered / highlighted metrics: {', '.join(selected_metrics)}",
            "",
        ])
        matching_cols = [c for c in leaderboard_df.columns if any(m in c for m in selected_metrics)]
        if matching_cols:
            md_lines.append(_df_to_markdown(leaderboard_df[matching_cols]))
            md_lines.append("")

    # Illumination breakdown table
    if "score" in df.columns:
        md_lines.extend([
            "## 2. Illumination Stratification Breakdown",
            "",
            "Stratified by illumination change score: **Same** (<0.2), **Moderate** (0.2 - 0.6), **Severe** (>=0.6).",
            "",
        ])
        rows = []
        for m, sub in df.groupby("model"):
            same_sub = sub[sub["score"] < 0.2]
            mod_sub = sub[(sub["score"] >= 0.2) & (sub["score"] < 0.6)]
            sev_sub = sub[sub["score"] >= 0.6]
            rows.append({
                "Model": m,
                "Same Sun RMSE (px)": same_sub["reproj_rmse_px"].mean(),
                "Moderate RMSE (px)": mod_sub["reproj_rmse_px"].mean(),
                "Severe RMSE (px)": sev_sub["reproj_rmse_px"].mean(),
                "Same Inliers": same_sub["num_inliers"].mean(),
                "Severe Inliers": sev_sub["num_inliers"].mean(),
                "Same SSIM": same_sub["reg_ssim"].mean(),
                "Severe SSIM": sev_sub["reg_ssim"].mean(),
            })
        illum_df = pd.DataFrame(rows).set_index("Model")
        md_lines.append(_df_to_markdown(illum_df))
        md_lines.append("")

    # Visualizations
    md_lines.extend([
        "## 3. Analytical Charts",
        "",
        "### Reprojection Error CDF",
        "![Reprojection RMSE CDF](rmse_cdf.png)",
        "",
        "### Inlier Ratio vs Spatial Coverage",
        "![Inlier Ratio vs Coverage](inlier_ratio_vs_coverage.png)",
        "",
        "### Runtime vs Reprojection Error",
        "![Runtime vs RMSE](runtime_vs_rmse.png)",
        "",
        "### Illumination Stratified Error",
        "![Illumination Stratified](illumination_stratified.png)",
        "",
    ])

    report_content = "\n".join(md_lines)
    report_path = out_dir / "report.md"
    report_path.write_text(report_content, encoding="utf-8")
    return report_content

def summarize(pairs_csv: str | Path, out_dir: str | Path, selected_metrics: Optional[List[str]] = None) -> str:
    """Aggregate per-pair CSV into leaderboard, plots, and markdown report."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pairs_csv)

    agg_dict = {k: v for k, v in AGG.items() if k in df.columns}
    lb = df.groupby("model").agg(agg_dict)

    # Flatten multi-level columns if any
    new_cols = []
    for col in lb.columns:
        if isinstance(col, tuple):
            a, b = col
            new_cols.append(f"{a}_{b}" if b else a)
        else:
            new_cols.append(col)
    lb.columns = new_cols

    # Save leaderboard.csv
    lb_path = out_path / "leaderboard.csv"
    lb.to_csv(lb_path)

    # Save summary.json
    summary_dict = lb.to_dict(orient="index")
    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=str)

    # Generate plots
    generate_plots(df, out_path)

    # Generate report.md
    build_markdown_report(lb, df, out_path, selected_metrics=selected_metrics)

    return str(lb_path)
