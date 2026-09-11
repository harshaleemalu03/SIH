import numpy as np
from registration.gt import reprojection_errors

def test_reprojection_errors_identity_is_zero():
    pred = np.array([[10.0, 20.0], [30.0, 40.0]])
    gt = np.array([[10.0, 20.0], [33.0, 44.0]])
    valid = np.array([True, True])
    errs = reprojection_errors(pred, gt, valid)
    assert errs.shape == (2,)
    assert abs(errs[0]) < 1e-6
    assert abs(errs[1] - 5.0) < 1e-6  # 3-4-5 triangle

def test_reprojection_errors_respects_valid_mask():
    pred = np.zeros((3, 2)); gt = np.ones((3, 2)); valid = np.array([True, False, True])
    assert reprojection_errors(pred, gt, valid).shape == (2,)
