#!/usr/bin/env python3
"""
Lunar Image Registration — FastAPI backend
Serves the minimal B/W UI and exposes /api/register for
source + reference image registration.
"""

from pathlib import Path
import base64
import io
import sys

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Lunar Image Registration")


# ------------------------------------------------------------------ helpers
def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image — unsupported or corrupt file.")
    return img


def _encode_b64(img: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _create_match_visualization(
    src_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    result: dict,
    matcher_name: str = "",
) -> np.ndarray:
    """Minimal B/W-friendly match vis — reuses logic from register_pair.py."""
    h, w = src_bgr.shape[:2]
    gap = 2  # thin divider
    canvas_h = h
    canvas_w = w * 2 + gap
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    canvas[:, :w] = src_bgr
    canvas[:, w + gap :] = ref_bgr
    # divider line
    canvas[:, w : w + gap] = 0

    kpts_src = result.get("matched_kpts0")
    kpts_ref = result.get("matched_kpts1")
    inlier_src = result.get("inlier_kpts0")

    inlier_indices: set[int] = set()
    if inlier_src is not None and kpts_src is not None:
        for j in range(len(inlier_src)):
            for i in range(len(kpts_src)):
                if np.allclose(kpts_src[i], inlier_src[j], atol=1e-2):
                    inlier_indices.add(i)
                    break

    n_total = len(kpts_src) if kpts_src is not None else 0
    indices = list(range(n_total))
    if n_total > 400:
        step = max(1, n_total // 400)
        indices = indices[::step]

    for i in indices:
        pt_src = kpts_src[i].astype(int)
        pt_ref = kpts_ref[i].astype(int) + np.array([w + gap, 0])
        is_inlier = i in inlier_indices
        # Yellow lines as requested: inlier = bright yellow, outlier = muted yellow
        color = (0, 255, 255) if is_inlier else (0, 180, 180)
        cv2.circle(canvas, tuple(pt_src), 2, color, -1)
        cv2.circle(canvas, tuple(pt_ref), 2, color, -1)
        cv2.line(canvas, tuple(pt_src), tuple(pt_ref), color, 1, cv2.LINE_AA)

    return canvas


# ------------------------------------------------------------------ routes
@app.get("/", include_in_schema=False)
def serve_index():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index)


@app.post("/api/register")
async def register(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
):
    # ---- validate & decode
    try:
        src_bytes = await source.read()
        ref_bytes = await reference.read()
        src_bgr = _decode_upload(src_bytes)
        ref_bgr = _decode_upload(ref_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    h_ref, w_ref = ref_bgr.shape[:2]

    # --- Primary: deep matcher (SuperPoint+LightGlue via vismatch) ---
    # Fallback: classic SIFT + BFMatcher + RANSAC (no torch needed)
    # This ensures the UI always runs a real matching + homography pipeline.
    H = None
    match_vis = None
    num_matches = 0
    num_inliers = 0
    inlier_ratio = 0.0
    used_fallback = False

    try:
        import torch
        from vismatch import get_matcher

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def to_tensor(bgr: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0

        src_rgb = to_tensor(src_bgr)
        ref_rgb = to_tensor(ref_bgr)

        model = get_matcher("superpoint-lightglue", device=device)
        result = model(src_rgb, ref_rgb)

        H = result.get("H")
        matched_kpts0 = result.get("matched_kpts0")
        num_matches = len(matched_kpts0) if matched_kpts0 is not None else 0
        num_inliers = int(result.get("num_inliers", 0))
        match_vis = _create_match_visualization(src_bgr, ref_bgr, result)
        inlier_ratio = round(num_inliers / max(1, num_matches) * 100, 1)
        print(f"[register] vismatch: matches={num_matches} inliers={num_inliers} H={'yes' if H is not None else 'no'}", file=sys.stderr)

    except Exception as e:
        print(f"[register] vismatch unavailable ({e}) — using SIFT fallback", file=sys.stderr)
        import traceback
        traceback.print_exc()
        used_fallback = True

    if used_fallback or match_vis is None:
        # ----- SIFT fallback (real pipeline, no torch) -----
        try:
            src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

            # SIFT — fallback to ORB if SIFT not available
            try:
                sift = cv2.SIFT_create(nfeatures=2000)
                norm = cv2.NORM_L2
            except AttributeError:
                sift = cv2.ORB_create(nfeatures=2000)
                norm = cv2.NORM_HAMMING

            kp1, des1 = sift.detectAndCompute(src_gray, None)
            kp2, des2 = sift.detectAndCompute(ref_gray, None)

            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                bf = cv2.BFMatcher(norm, crossCheck=False)
                # Lowe ratio test
                knn = bf.knnMatch(des1, des2, k=2)
                good = []
                for m, n in knn:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

                # Cap for visualization speed
                good_sorted = sorted(good, key=lambda x: x.distance)[:500]
                num_matches = len(good_sorted)

                if len(good_sorted) >= 4:
                    pts_src = np.float32([kp1[m.queryIdx].pt for m in good_sorted])
                    pts_ref = np.float32([kp2[m.trainIdx].pt for m in good_sorted])
                    H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 4.0)
                    if mask is not None:
                        num_inliers = int(mask.sum())
                        inlier_ratio = round(num_inliers / max(1, num_matches) * 100, 1)
                    else:
                        num_inliers = 0
                    # Build result dict for visualization
                    result_fb = {
                        "matched_kpts0": pts_src,
                        "matched_kpts1": pts_ref,
                        "inlier_kpts0": pts_src[mask.ravel() == 1] if mask is not None else np.zeros((0, 2)),
                        "num_inliers": num_inliers,
                    }
                    match_vis = _create_match_visualization(src_bgr, ref_bgr, result_fb)
                else:
                    match_vis = _create_match_visualization(
                        src_bgr, ref_bgr, {"matched_kpts0": np.zeros((0,2)), "matched_kpts1": np.zeros((0,2)), "inlier_kpts0": np.zeros((0,2))}
                    )
                    H = None
            else:
                match_vis = _create_match_visualization(
                    src_bgr, ref_bgr, {"matched_kpts0": np.zeros((0,2)), "matched_kpts1": np.zeros((0,2)), "inlier_kpts0": np.zeros((0,2))}
                )
                H = None
                num_matches = 0
                num_inliers = 0
        except Exception as e2:
            print(f"[register] SIFT fallback failed: {e2}", file=sys.stderr)
            traceback.print_exc()
            # Last-resort: side-by-side without lines
            match_vis = np.ones((max(src_bgr.shape[0], ref_bgr.shape[0]), src_bgr.shape[1] + ref_bgr.shape[1] + 2, 3), dtype=np.uint8) * 255
            match_vis[0 : src_bgr.shape[0], 0 : src_bgr.shape[1]] = src_bgr
            match_vis[0 : ref_bgr.shape[0], src_bgr.shape[1] + 2 :] = ref_bgr
            match_vis[:, src_bgr.shape[1] : src_bgr.shape[1] + 2] = 0
            H = None

    # Warp source to reference frame + compute MSE
    mse = None
    if H is not None:
        warped = cv2.warpPerspective(src_bgr, H, (w_ref, h_ref))
        # MSE over valid warped region (mask where source maps)
        mask = cv2.warpPerspective(np.ones((src_bgr.shape[0], src_bgr.shape[1]), dtype=np.uint8) * 255, H, (w_ref, h_ref))
        valid = mask > 0
        if np.any(valid):
            diff = warped.astype(np.float32) - ref_bgr.astype(np.float32)
            mse = float(np.mean((diff[valid]) ** 2))
        else:
            diff = warped.astype(np.float32) - ref_bgr.astype(np.float32)
            mse = float(np.mean(diff ** 2))
    else:
        warped = src_bgr
        # If no homography, MSE against resized reference if sizes differ — else direct
        if src_bgr.shape[:2] == ref_bgr.shape[:2]:
            diff = src_bgr.astype(np.float32) - ref_bgr.astype(np.float32)
            mse = float(np.mean(diff ** 2))
        else:
            # resize source to reference for a comparable MSE
            resized = cv2.resize(src_bgr, (w_ref, h_ref))
            diff = resized.astype(np.float32) - ref_bgr.astype(np.float32)
            mse = float(np.mean(diff ** 2))

    return JSONResponse(
        {
            "match_image": "data:image/jpeg;base64," + _encode_b64(match_vis),
            "registered_image": "data:image/jpeg;base64," + _encode_b64(warped),
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": inlier_ratio,
            "mse": round(mse, 2) if mse is not None else None,
            "homography_found": H is not None,
        }
    )


# Mount static after routes so "/" takes precedence
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
