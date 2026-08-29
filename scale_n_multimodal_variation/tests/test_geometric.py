import cv2
import numpy as np

from src.registration.geometric import estimate_homography


def test_homography_estimation():

    keypoints1 = [
        cv2.KeyPoint(0, 0, 1),
        cv2.KeyPoint(100, 0, 1),
        cv2.KeyPoint(100, 100, 1),
        cv2.KeyPoint(0, 100, 1),
    ]

    keypoints2 = [
        cv2.KeyPoint(10, 20, 1),
        cv2.KeyPoint(110, 20, 1),
        cv2.KeyPoint(110, 120, 1),
        cv2.KeyPoint(10, 120, 1),
    ]

    matches = []

    for i in range(4):
        matches.append(
            cv2.DMatch(i, i, 0)
        )

    H, mask = estimate_homography(
        keypoints1,
        keypoints2,
        matches
    )

    assert H is not None
    assert mask is not None
    assert int(mask.sum()) >= 4
