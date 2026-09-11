import numpy as np
from registration.adapters import MatchOutput
from registration.metrics import compute_pair_metrics

def _match(n=100, k=50):
    return MatchOutput(
        matched_kpts0=np.zeros((n, 2)), matched_kpts1=np.zeros((n, 2)),
        inlier_kpts0=np.zeros((k, 2)), inlier_kpts1=np.zeros((k, 2)),
        H=np.eye(3) if k >= 4 else None, num_candidates=n, num_inliers=k,
        inlier_ratio=k / n if n > 0 else 0.0, latency_match_ms=10.0, latency_ransac_ms=2.0)

def test_counts_and_ratio():
    m = compute_pair_metrics(_match(100, 50), np.array([1.0, 2.0, 3.0]), (512, 512), 12.0, None)
    assert (m.num_candidates, m.num_inliers) == (100, 50)
    assert abs(m.inlier_ratio - 0.5) < 1e-9
    assert abs(m.reproj_rmse_px - np.sqrt((1 + 4 + 9) / 3)) < 1e-9
    assert abs(m.reproj_median_px - 2.0) < 1e-9 and abs(m.reproj_max_px - 3.0) < 1e-9

def test_uniformity_and_coverage_grids():
    pts = np.array([[128 * (i % 4) + 64.0, 128 * (i // 4 % 4) + 64.0] for i in range(16)])
    m = MatchOutput(matched_kpts0=pts, matched_kpts1=pts, inlier_kpts0=pts,
                    inlier_kpts1=pts, H=np.eye(3), num_candidates=16,
                    num_inliers=16, inlier_ratio=1.0)
    out = compute_pair_metrics(m, np.ones(16), (512, 512), 1.0, None)
    assert abs(out.uniformity_score - 1.0) < 1e-9
    assert abs(out.spatial_coverage - 1.0) < 1e-9

def test_empty_errors_give_nan():
    out = compute_pair_metrics(_match(0, 0), np.array([]), (512, 512), 0.0, None)
    assert np.isnan(out.reproj_rmse_px) and out.uniformity_score == 0.0 and out.spatial_coverage == 0.0
