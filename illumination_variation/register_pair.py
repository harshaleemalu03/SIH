#!/usr/bin/env python3
"""
Lunar Image Registration & Match Visualization CLI
==================================================
Runs deep feature matching (SuperPoint + LightGlue, SIFT + LightGlue, LoFTR, etc.)
on any arbitrary source and reference image pair, computes homography alignment,
and exports presentation-ready visual outputs.

Usage:
    python register_pair.py --source path/to/source.jpg --reference path/to/reference.jpg
    python register_pair.py -s img0.jpg -r img1.jpg -o output_dir --matcher superpoint-lightglue
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vismatch import get_matcher, available_models

app = typer.Typer(help="Run keypoint matching and registration on any image pair.")
console = Console()


def create_match_visualization(
    src_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    result: dict,
    matcher_name: str,
    src_label: str = "Source Image",
    ref_label: str = "Reference Image",
) -> np.ndarray:
    """Renders a side-by-side keypoint match visualization with inlier/outlier lines."""
    h, w = src_bgr.shape[:2]
    gap = 30
    header_h = 40
    canvas_h = h + header_h
    canvas_w = w * 2 + gap
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30

    canvas[header_h:, :w] = src_bgr
    canvas[header_h:, w + gap :] = ref_bgr

    kpts_src = result.get("matched_kpts0")
    kpts_ref = result.get("matched_kpts1")
    inlier_src = result.get("inlier_kpts0")
    num_inliers = result.get("num_inliers", 0)

    inlier_indices = set()
    if inlier_src is not None and kpts_src is not None:
        for j in range(len(inlier_src)):
            for i in range(len(kpts_src)):
                if np.allclose(kpts_src[i], inlier_src[j], atol=1e-2):
                    inlier_indices.add(i)
                    break

    n_total = len(kpts_src) if kpts_src is not None else 0
    indices = list(range(n_total))
    if n_total > 300:
        step = max(1, n_total // 300)
        indices = indices[::step]

    for i in indices:
        pt_src = kpts_src[i].astype(int) + np.array([0, header_h])
        pt_ref = kpts_ref[i].astype(int) + np.array([w + gap, header_h])

        is_inlier = (i in inlier_indices)
        color = (0, 230, 0) if is_inlier else (0, 0, 220)

        cv2.circle(canvas, tuple(pt_src), 3, color, -1)
        cv2.circle(canvas, tuple(pt_ref), 3, color, -1)
        cv2.line(canvas, tuple(pt_src), tuple(pt_ref), color, 1, cv2.LINE_AA)

    # Title texts
    cv2.putText(canvas, src_label, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, ref_label, (w + gap + 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    inlier_pct = (num_inliers / max(1, n_total)) * 100
    info_text = f"{matcher_name} | Matches: {n_total} | Inliers: {num_inliers} ({inlier_pct:.1f}%)"
    cv2.putText(canvas, info_text, (15, canvas_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas


@app.command()
def main(
    source: Path = typer.Option(..., "--source", "-s", help="Path to input source image"),
    reference: Path = typer.Option(..., "--reference", "-r", help="Path to input reference image"),
    output_dir: Path = typer.Option(Path("output_registered_pair"), "--output-dir", "-o", help="Output directory to save images"),
    matcher: str = typer.Option("superpoint-lightglue", "--matcher", "-m", help="Matcher model (e.g. superpoint-lightglue, sift-lightglue, loftr)"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device to run on ('cuda' or 'cpu')"),
):
    """Aligns source image to reference image using deep matching and homography registration."""
    
    if not source.exists():
        console.print(f"[bold red]Error:[/] Source image not found: {source}")
        raise typer.Exit(code=1)
    if not reference.exists():
        console.print(f"[bold red]Error:[/] Reference image not found: {reference}")
        raise typer.Exit(code=1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(f"[bold cyan]Source:[/] {source}\n[bold cyan]Reference:[/] {reference}\n[bold cyan]Matcher:[/] {matcher}\n[bold cyan]Device:[/] {device}\n[bold cyan]Output:[/] {output_dir}", title="Image Registration"))

    # Load images
    src_bgr = cv2.imread(str(source))
    ref_bgr = cv2.imread(str(reference))

    if src_bgr is None:
        console.print(f"[bold red]Error:[/] Failed to decode source image {source}")
        raise typer.Exit(code=1)
    if ref_bgr is None:
        console.print(f"[bold red]Error:[/] Failed to decode reference image {reference}")
        raise typer.Exit(code=1)

    # Save original inputs
    cv2.imwrite(str(output_dir / "source_image.jpg"), src_bgr)
    cv2.imwrite(str(output_dir / "reference_image.jpg"), ref_bgr)

    # Convert to RGB (C, H, W) normalized float32
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0

    with console.status(f"[bold green]Running {matcher}..."):
        model = get_matcher(matcher, device=device)
        result = model(src_rgb, ref_rgb)

    H = result.get("H")
    num_inliers = result.get("num_inliers", 0)
    matched_kpts0 = result.get("matched_kpts0")
    num_matches = len(matched_kpts0) if matched_kpts0 is not None else 0

    h, w = src_bgr.shape[:2]

    # 1. Match Point View
    match_vis = create_match_visualization(
        src_bgr,
        ref_bgr,
        result,
        matcher,
        src_label=f"Source ({source.name})",
        ref_label=f"Reference ({reference.name})",
    )
    cv2.imwrite(str(output_dir / "match_point_view.jpg"), match_vis)

    # 2. Homography Warping / Registration
    if H is not None:
        warped_src = cv2.warpPerspective(src_bgr, H, (w, h))
        cv2.imwrite(str(output_dir / "registered_image.jpg"), warped_src)

        # 3. Alpha Blended Overlay
        mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, H, (w, h))
        overlay = ref_bgr.copy()
        valid = mask > 0
        overlay[valid] = cv2.addWeighted(ref_bgr[valid], 0.5, warped_src[valid], 0.5, 0)
        cv2.imwrite(str(output_dir / "alignment_overlay.jpg"), overlay)

        # 4. Checkerboard Comparison
        checker = np.zeros_like(ref_bgr)
        block_size = max(16, min(h, w) // 8)
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if ((y // block_size) + (x // block_size)) % 2 == 0:
                    checker[y : min(y + block_size, h), x : min(x + block_size, w)] = warped_src[
                        y : min(y + block_size, h), x : min(x + block_size, w)
                    ]
                else:
                    checker[y : min(y + block_size, h), x : min(x + block_size, w)] = ref_bgr[
                        y : min(y + block_size, h), x : min(x + block_size, w)
                    ]
        cv2.imwrite(str(output_dir / "alignment_checkerboard.jpg"), checker)

        # 5. 2x2 Presentation Grid Summary
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor="#1e1e1e")

        axes[0, 0].imshow(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"1. Source Image ({source.name})", color="white", fontsize=14, pad=10)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"2. Reference Image ({reference.name})", color="white", fontsize=14, pad=10)
        axes[0, 1].axis("off")

        axes[1, 0].imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"3. {matcher} ({num_inliers} Inliers)", color="white", fontsize=14, pad=10)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(cv2.cvtColor(warped_src, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("4. Registered Output (Aligned Source)", color="white", fontsize=14, pad=10)
        axes[1, 1].axis("off")

        plt.suptitle("Lunar Feature Matching & Registration Pipeline", color="white", fontsize=16, y=0.98, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(output_dir / "pair_summary_grid.jpg"), dpi=200, bbox_inches="tight", facecolor="#1e1e1e")
        plt.close()

    # Results summary table
    table = Table(title="Registration Results", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total Matches", str(num_matches))
    table.add_row("Inliers", str(num_inliers))
    table.add_row("Inlier Ratio", f"{(num_inliers / max(1, num_matches)) * 100:.1f}%")
    table.add_row("Homography Found", "[green]Yes[/]" if H is not None else "[red]No[/]")
    console.print(table)
    console.print(f"[bold green]✓ Done![/] All outputs saved to [bold cyan]{output_dir.resolve()}[/]")


if __name__ == "__main__":
    app()
