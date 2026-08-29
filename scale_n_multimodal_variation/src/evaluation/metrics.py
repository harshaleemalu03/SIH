import cv2
import numpy as np

def inlier_count(mask):
    """Return the number of RANSAC inliers."""

    if mask is None:
        return 0

    return int(np.sum(mask))


def inlier_ratio(mask):
    """Return the fraction of matches classified as inliers."""

    if mask is None:
        return 0.0

    mask = np.asarray(mask).ravel()

    if len(mask) == 0:
        return 0.0

    return float(np.mean(mask))


def reprojection_rmse(keypoints1, keypoints2, matches, homography, mask):
    """
    Calculate RMSE of the inlier reprojection errors.
    """

    if homography is None or mask is None:
        return None

    mask = np.asarray(mask).ravel().astype(bool)

    if not np.any(mask):
        return None

    source_points = np.float32([
        keypoints1[m.queryIdx].pt
        for m in matches
    ])

    target_points = np.float32([
        keypoints2[m.trainIdx].pt
        for m in matches
    ])

    projected = cv2.perspectiveTransform(
        source_points.reshape(-1, 1, 2),
        homography
    ).reshape(-1, 2)

    errors = projected[mask] - target_points[mask]

    squared_errors = np.sum(errors ** 2, axis=1)

    return float(np.sqrt(np.mean(squared_errors)))
