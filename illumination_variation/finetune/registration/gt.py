"""Ground-truth reprojection targets. WHY: lunar terrain is non-planar, so a
single GT homography is wrong; per-keypoint depth-warped GT (warp_kpts) is the
only geometrically valid reference. Reuses EfficientLoFTR's warp_kpts."""
import numpy as np
import torch

def gt_reprojections(kpts0, depth0, depth1, T_0to1, K0, K1):
    """Warp kpts0 into image-1 GT positions. Returns (valid_mask (M,) bool
    numpy, gt_pts1 (M,2) numpy, invalid entries NaN)."""
    import sys
    from pathlib import Path
    if 'src.loftr.utils.geometry' not in sys.modules:
        proj = Path(__file__).resolve().parent.parent / "EfficientLoFTR"
        if str(proj) not in sys.path:
            sys.path.insert(0, str(proj))
    from src.loftr.utils.geometry import warp_kpts
    k_arr = np.asarray(kpts0, dtype=np.float32).reshape(-1, 2)
    if len(k_arr) == 0:
        return np.zeros(0, dtype=bool), np.zeros((0, 2))
    h, w = (depth0.shape[-2], depth0.shape[-1]) if hasattr(depth0, "shape") else (512, 512)
    k_arr[:, 0] = np.clip(k_arr[:, 0], 0.0, float(w - 1.001))
    k_arr[:, 1] = np.clip(k_arr[:, 1], 0.0, float(h - 1.001))

    k = torch.as_tensor(k_arr).unsqueeze(0)
    dev = torch.as_tensor(depth0).device if torch.is_tensor(depth0) else torch.device("cpu")
    k = k.to(dev)
    d0 = torch.as_tensor(depth0).unsqueeze(0).to(dev)
    d1 = torch.as_tensor(depth1).unsqueeze(0).to(dev)
    T = torch.as_tensor(T_0to1).unsqueeze(0).to(dev)
    K_0 = torch.as_tensor(K0).unsqueeze(0).to(dev)
    K_1 = torch.as_tensor(K1).unsqueeze(0).to(dev)
    with torch.inference_mode():
        valid, warped = warp_kpts(k, d0, d1, T, K_0, K_1)
    valid_np = valid[0].detach().cpu().numpy().astype(bool)
    gt_np = warped[0].detach().cpu().numpy()
    gt_np[~valid_np] = np.nan
    return valid_np, gt_np

def reprojection_errors(pred_pts1, gt_pts1, valid_mask):
    """Euclidean px error per valid correspondence."""
    pred = np.asarray(pred_pts1, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_pts1, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(valid_mask).reshape(-1).astype(bool)
    pv, gv = pred[m], gt[m]
    ok = np.isfinite(gv).all(axis=1)
    return np.linalg.norm(pv[ok] - gv[ok], axis=1)
