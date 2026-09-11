import pandas as pd
from registration.report import summarize

def test_summarize_aggregates_ten_metrics(tmp_path):
    df = pd.DataFrame([{
        "model": "test_model",
        "num_candidates": 100,
        "num_inliers": 50,
        "inlier_ratio": 0.5,
        "reproj_rmse_px": 2.0,
        "reproj_median_px": 1.5,
        "reproj_max_px": 5.0,
        "uniformity_score": 0.8,
        "spatial_coverage": 0.9,
        "runtime_ms": 100.0,
        "reg_mse": 1.0,
        "reg_psnr_db": 30.0,
        "reg_ssim": 0.9,
        "reg_ncc": 0.95,
        "overlap_ratio": 0.8,
        "score": 0.1,
    }])
    p = tmp_path / "pairs.csv"
    df.to_csv(p, index=False)
    out = summarize(str(p), str(tmp_path))
    assert (tmp_path / "leaderboard.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "rmse_cdf.png").exists()
    assert (tmp_path / "inlier_ratio_vs_coverage.png").exists()
    assert (tmp_path / "runtime_vs_rmse.png").exists()
    assert (tmp_path / "illumination_stratified.png").exists()
    content = open(tmp_path / "leaderboard.csv").read()
    assert "reproj_rmse_px_mean" in content
