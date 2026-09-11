import numpy as np
from registration.adapters import finalize_output
from registration.ransac import run_ransac as rr

def test_vismatch_raw_matches_are_refiltered_uniformly():
    rng = np.random.default_rng(2)
    m0 = rng.uniform(0, 512, size=(50, 2))
    m1 = m0 + np.array([5.0, 0.0])
    H, i0, i1, mask = rr(m0, m1)
    out = finalize_output(m0, m1, H, i0, i1, mask, t_match=5.0, t_ransac=1.0)
    assert out.num_candidates == 50 and out.inlier_ratio > 0.8
    assert abs(out.latency_match_ms - 5.0) < 1e-9

def test_normalize_model_name():
    from registration.adapters import normalize_model_name
    assert normalize_model_name("lightglue") == "superpoint-lightglue"
    assert normalize_model_name("eloftr") == "eloftr_finetuned"
    assert normalize_model_name("eloftr_base") == "eloftr_base"
    assert normalize_model_name("roma") == "roma"
    assert normalize_model_name("superglue") == "superglue"
    assert normalize_model_name("loftr") == "loftr"
