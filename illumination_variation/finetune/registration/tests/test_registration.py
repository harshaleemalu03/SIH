import numpy as np
from registration.registration import warp_and_score

def test_identity_warp_is_perfect():
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    out = warp_and_score(ref.copy(), ref, np.eye(3))
    assert out["overlap_ratio"] > 0.99
    assert out["mse"] < 1e-9 and out["psnr_db"] > 80
    assert out["ssim"] > 0.99 and out["ncc"] > 0.99

def test_none_H_gives_nan():
    ref = np.zeros((64, 64, 3), dtype=np.uint8)
    out = warp_and_score(ref, ref, None)
    assert out["overlap_ratio"] == 0.0 and np.isnan(out["mse"])
