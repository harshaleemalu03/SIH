import numpy as np
from registration.ransac import run_ransac, RANSACConfig, RANSAC_THRESH_PX

def test_run_ransac_recovers_translation_with_outliers():
    rng = np.random.default_rng(0)
    src = rng.uniform(50, 450, size=(60, 2)).astype(np.float64)
    dst = src + np.array([20.0, -10.0])  # pure translation
    dst[:15] += rng.uniform(-200, 200, size=(15, 2))  # 25% outliers
    H, in0, in1, mask = run_ransac(src, dst, RANSACConfig())
    assert H is not None
    assert mask.sum() >= 40
    assert abs(H[0, 2] - 20.0) < 2.0 and abs(H[1, 2] + 10.0) < 2.0

def test_run_ransac_fewer_than_4_returns_none():
    H, in0, in1, mask = run_ransac(np.zeros((3, 2)), np.zeros((3, 2)), RANSACConfig())
    assert H is None and len(in0) == 0 and mask.sum() == 0

def test_default_thresh_is_3px():
    assert RANSAC_THRESH_PX == 3.0
