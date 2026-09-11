"""Single shared RANSAC stage. WHY: every matcher must face identical geometry
filtering, otherwise inlier counts are incomparable (vismatch filters, eLoFTR did not)."""
from dataclasses import dataclass
import cv2
import numpy as np

RANSAC_THRESH_PX = 3.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.995

@dataclass(frozen=True)
class RANSACConfig:
    """Exact OpenCV params shared by ALL methods."""
    reproj_thresh_px: float = RANSAC_THRESH_PX
    max_iters: int = RANSAC_MAX_ITERS
    confidence: float = RANSAC_CONFIDENCE

def run_ransac(mkpts0, mkpts1, config=RANSACConfig()):
    """Filter candidate matches with USAC_MAGSAC homography.
    Returns (H|None, inlier_kpts0, inlier_kpts1, inlier_mask bool[N])."""
    mkpts0 = np.asarray(mkpts0, dtype=np.float64).reshape(-1, 2)
    mkpts1 = np.asarray(mkpts1, dtype=np.float64).reshape(-1, 2)
    n = len(mkpts0)
    if n < 4 or len(mkpts1) < 4:
        return None, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(n, dtype=bool)
    H, mask = cv2.findHomography(
        mkpts0, mkpts1,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=config.reproj_thresh_px,
        maxIters=config.max_iters,
        confidence=config.confidence,
    )
    if H is None or mask is None:
        return None, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(n, dtype=bool)
    m = mask.reshape(-1).astype(bool)
    return H, mkpts0[m], mkpts1[m], m
