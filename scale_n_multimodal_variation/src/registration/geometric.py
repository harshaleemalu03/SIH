import cv2
import numpy as np


def estimate_homography(keypoints1, keypoints2, matches, threshold=5.0):
    """
    Estimate a homography using RANSAC and identify inlier matches.
    """

    if len(matches) < 4:
        return None, None

    points1 = np.float32([
        keypoints1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    points2 = np.float32([
        keypoints2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        threshold
    )

    return H, mask
