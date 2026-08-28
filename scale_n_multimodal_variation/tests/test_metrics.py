import numpy as np

from src.evaluation.metrics import (
    inlier_count,
    inlier_ratio
)


def test_inlier_metrics():

    mask = np.array([
        [1],
        [1],
        [0],
        [1]
    ])

    assert inlier_count(mask) == 3
    assert inlier_ratio(mask) == 0.75
