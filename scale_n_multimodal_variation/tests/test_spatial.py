import cv2
import numpy as np

from src.evaluation.spatial import match_distribution


def test_match_distribution():

    image_shape = (400, 400)

    keypoints = [
        cv2.KeyPoint(50, 50, 1),
        cv2.KeyPoint(150, 50, 1),
        cv2.KeyPoint(250, 50, 1),
        cv2.KeyPoint(350, 50, 1),
        cv2.KeyPoint(50, 150, 1),
        cv2.KeyPoint(150, 150, 1),
        cv2.KeyPoint(250, 150, 1),
        cv2.KeyPoint(350, 150, 1),
    ]

    matches = [
        cv2.DMatch(i, i, 0)
        for i in range(len(keypoints))
    ]

    coverage, counts, uniformity = match_distribution(
        keypoints,
        matches,
        image_shape,
        grid_rows=2,
        grid_cols=2
    )

    assert coverage > 0
    assert counts.sum() == len(matches)
    assert 0 <= uniformity <= 1
