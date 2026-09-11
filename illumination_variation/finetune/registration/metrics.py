"""The 10 requested metrics. WHY one module: guarantees every method reports
identical columns with identical edge-case handling (NaN when undefined)."""
from dataclasses import dataclass, field
import numpy as np

GRID = 4  # 4x4 = 16 cells over the 512x512 source frame
METRIC_COLUMNS = [
    "num_candidates",
    "num_inliers",
    "inlier_ratio",
    "reproj_rmse_px",
    "reproj_median_px",
    "reproj_max_px",
    "uniformity_score",
    "spatial_coverage",
    "runtime_ms",
    "reg_mse",
    "reg_psnr_db",
    "reg_ssim",
    "reg_ncc",
    "overlap_ratio",
]

@dataclass
class PairMetrics:
    num_candidates: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reproj_rmse_px: float = float("nan")
    reproj_median_px: float = float("nan")
    reproj_max_px: float = float("nan")
    uniformity_score: float = 0.0
    spatial_coverage: float = 0.0
    runtime_ms: float = 0.0
    registered_quality: dict = field(default_factory=dict)

def _grid_stats(inlier_pts, img_shape, grid=GRID):
    h, w = img_shape[:2]
    if inlier_pts is None or len(inlier_pts) == 0:
        return 0.0, 0.0
    xs = np.clip((np.asarray(inlier_pts)[:, 0] / w * grid).astype(int), 0, grid - 1)
    ys = np.clip((np.asarray(inlier_pts)[:, 1] / h * grid).astype(int), 0, grid - 1)
    counts = np.zeros((grid, grid))
    np.add.at(counts, (ys, xs), 1)
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0
    prob = counts.reshape(-1) / total
    nz = prob[prob > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    uniformity = entropy / float(np.log(grid * grid)) if total > 0 else 0.0
    coverage = float((counts > 0).sum() / (grid * grid))
    return float(uniformity), float(coverage)

def compute_pair_metrics(match, errors, img_shape, runtime_ms, quality):
    errs = np.asarray(errors, dtype=np.float64).reshape(-1)
    errs = errs[np.isfinite(errs)]
    if len(errs) > 0:
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        med = float(np.median(errs))
        mx = float(np.max(errs))
    else:
        rmse = med = mx = float("nan")
    uni, cov = _grid_stats(match.inlier_kpts0, img_shape)
    q = quality or {}
    return PairMetrics(
        num_candidates=int(match.num_candidates),
        num_inliers=int(match.num_inliers),
        inlier_ratio=float(match.inlier_ratio),
        reproj_rmse_px=rmse,
        reproj_median_px=med,
        reproj_max_px=mx,
        uniformity_score=uni,
        spatial_coverage=cov,
        runtime_ms=float(runtime_ms),
        registered_quality={
            "mse": float(q.get("mse", float("nan"))),
            "psnr_db": float(q.get("psnr_db", float("nan"))),
            "ssim": float(q.get("ssim", float("nan"))),
            "ncc": float(q.get("ncc", float("nan"))),
            "overlap_ratio": float(q.get("overlap_ratio", 0.0)),
        },
    )
