from registration.metrics import METRIC_COLUMNS

def test_metric_columns_cover_ten_requests():
    cols = set(METRIC_COLUMNS)
    for required in [
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
    ]:
        assert required in cols

def test_evaluate_pair_importable():
    from registration.benchmark import evaluate_pair, build_adapter  # noqa
