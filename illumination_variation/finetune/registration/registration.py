"""Warp + registered-image quality. WHY masked photometrics: full-frame MSE
punishes valid non-overlap black borders; only overlap pixels count."""
import cv2
import numpy as np

def warp_and_score(src_bgr, ref_bgr, H):
    h, w = ref_bgr.shape[:2]
    if H is None:
        return {
            "mse": float("nan"),
            "psnr_db": float("nan"),
            "ssim": float("nan"),
            "ncc": float("nan"),
            "overlap_ratio": 0.0,
            "warped": None,
            "overlap_mask": np.zeros((h, w), dtype=bool),
        }
    H = np.asarray(H, dtype=np.float64)
    warped = cv2.warpPerspective(src_bgr, H, (w, h))
    mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, H, (w, h)) > 0
    ratio = float(mask.mean())
    if ratio < 0.01:
        return {
            "mse": float("nan"),
            "psnr_db": float("nan"),
            "ssim": float("nan"),
            "ncc": float("nan"),
            "overlap_ratio": ratio,
            "warped": warped,
            "overlap_mask": mask,
        }
    a = warped[mask].astype(np.float64)
    b = ref_bgr[mask].astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    psnr = float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")
    wg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ssim = float(ssim_fn(rg, wg, data_range=255))
    except Exception:
        ssim = float("nan")
    x = wg[mask].astype(np.float64).ravel()
    y = rg[mask].astype(np.float64).ravel()
    xz, yz = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xz ** 2).sum() * (yz ** 2).sum()))
    ncc = float((xz * yz).sum() / denom) if denom > 0 else float("nan")
    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim,
        "ncc": ncc,
        "overlap_ratio": ratio,
        "warped": warped,
        "overlap_mask": mask,
    }
