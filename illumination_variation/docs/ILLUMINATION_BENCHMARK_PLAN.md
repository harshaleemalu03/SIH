# Hardcore Illumination Benchmark Plan (v2)

## Critical Findings from Dataset Analysis

### What We Actually Have

The dataset is **StereoLunar** (not the StereoGeo calibration paper). Per the ICCV 2025 paper and GitHub README:

| File | Content | Format |
|------|---------|--------|
| `.jpg` | RGB image | 512×512, 8-bit |
| `.exr` | **Depth map** (metric scale) | Float32, meters |
| `.npz` | Camera intrinsics + cam2world | 3×3 + 4×4 matrices |

**Key realization:** The `.exr` files are **depth maps**, not radiance. This means we have full 3D supervision.

### Solar Angles (from the paper)

The dataset uses three lighting conditions per trajectory:
- **Side lighting**: azimuth=150°, incidence=160°
- **Overhead lighting**: azimuth=250°, incidence=20°
- **Back lighting**: azimuth=360°, incidence=165°

These are NOT stored in the `.npz` files, but are known from the rendering setup. For our benchmark, we can:
1. Compute the **azimuth difference** between consecutive pairs in the same trajectory
2. Compute **shadow fraction** from the depth maps + known sun direction
3. Use **radiance ratio** as a proxy for illumination change (but this is what you correctly identified as insufficient)

### What the Full Dataset Provides (50,000+ pairs)

From the Zenodo page:
- Dense depth maps (per-pixel)
- Camera-to-world poses in Moon-fixed Frame
- Stereo baseline (3D inter-camera displacement)
- Georeferenced metadata: absolute altitude, GSD, trajectory metadata
- Three lighting conditions per trajectory

---

## Corrected Plan

### Phase 0: GT Validation (Before Anything Else)

**Why:** Your #1 criticism. ORB+RANSAC as GT is circular. We must validate using dataset-native data.

**Task 0.1: Verify depth map quality**
- Load `.exr` files, check for NaN/inf values
- Visualize depth maps alongside RGB images
- Compute depth statistics (min, max, mean, std) per split
- Output: `depth_quality_report.json`

**Task 0.2: Compute GT correspondences from camera poses + depth**
```python
def compute_gt_correspondence(img_a, depth_a, K_a, cam2world_a, 
                               K_b, cam2world_b, pixel_a):
    """Project pixel_a from image A to image B using depth + poses."""
    # 1. Unproject pixel_a to 3D using depth_a and K_a
    z = depth_a[pixel_a[1], pixel_a[0]]
    p3d = K_a_inv @ np.array([pixel_a[0], pixel_a[1], 1]) * z
    
    # 2. Transform to world coordinates
    p3d_world = cam2world_a @ np.append(p3d, 1)
    
    # 3. Project to image B using K_b and cam2world_b
    p2d_b = K_b @ cam2world_b_inv @ p3d_world
    pixel_b = p2d_b[:2] / p2d_b[2]
    
    return pixel_b
```

**Task 0.3: Validate GT on 20 pairs**
- Run the above on 20 random pairs (5 per split)
- Check if GT correspondences are plausible (within image bounds, reasonable displacement)
- Output: `gt_validation_20_pairs.png` (visual inspection)

**Task 0.4: Handle non-planar terrain**
- For each pair, compute **relief metric**: std(depth) / mean(depth)
- If relief > threshold, flag as "non-planar" and evaluate with 3D reprojection error instead of H_error
- Output: `terrain_relief_distribution.png`

---

### Phase 1: Dataset Distribution Analysis

**Why:** Your #6 criticism. Check if proposed bins have enough pairs before building infrastructure.

**Task 1.1: Compute illumination parameters per pair**
- For consecutive pairs in same trajectory, compute:
  - **Azimuth difference**: Δ azimuth (if known from trajectory metadata)
  - **Incidence angle**: from sun direction (if available)
  - **Shadow fraction**: fraction of pixels with depth > threshold (proxy for shadowed area)
  - **Radiance ratio**: mean(JPG_b) / mean(JPG_a) (simple proxy)

**Task 1.2: Bin distribution analysis**
- Create 2D grid: incidence angle × azimuth difference
- Count pairs per bin
- Check if bins have ≥10 pairs (minimum for statistical significance)
- Output: `bin_distribution_heatmap.png`

**Task 1.3: Shadow fraction analysis**
- For each pair, compute shadow fraction from depth maps
- Shadow pixels: depth > median_depth + 2*std(depth)
- Output: `shadow_fraction_distribution.png`

---

### Phase 2: Stage-by-Stage Metrics (with corrected GT)

**Why:** Your #1 and #3 criticisms. Use 3D correspondences, not homographies.

**Task 2.1: Detection Repeatability (corrected)**
- For each keypoint in image A, warp to 3D using depth, project to image B
- Check if a keypoint exists within ε pixels in image B
- Output: `repeatability = matches_with_keypoint / total_keypoints_A`
- **Reveals:** Does the detector find the same physical points?

**Task 2.2: Descriptor Invariance (matcher-aware)**
- For matched keypoints (after 3D warping), compute descriptor distance
- Use appropriate metric per matcher:
  - SuperPoint/LightGlue: L2 distance
  - ORB: Hamming distance
  - LoFTR: Learned descriptor distance
- Output: `descriptor_distance` distribution per matcher
- **Reveals:** Do descriptors stay similar under illumination change?

**Task 2.3: Matching Consistency**
- For each putative match, check if it's within ε pixels of GT correspondence
- Output: `precision`, `recall`, `F1`
- **Reveals:** Are the matches correct?

**Task 2.4: Geometric Verification (corrected)**
- For planar pairs: Compute H_pred, compare with GT H (from 3D correspondences)
- For non-planar pairs: Compute **reprojection error** using depth + poses
- Output: `H_error` (planar) or `reproj_error` (non-planar)
- **Reveals:** Is the final registration accurate?

---

### Phase 3: Illumination-Stratified Analysis

**Why:** Your #2 criticism. Use physical parameters, not pixel ratios.

**Task 3.1: Stratify by shadow fraction**
- Bin pairs by shadow fraction: (0-10%), (10-20%), (20-30%), (30-40%), (40-50%), (50%+)
- For each bin, compute all stage metrics
- Output: Performance curve vs shadow fraction

**Task 3.2: Stratify by terrain relief**
- Bin pairs by relief metric: (0-0.1), (0.1-0.2), (0.2-0.3), (0.3+)
- For each bin, compute all stage metrics
- Output: Performance curve vs terrain relief

**Task 3.3: Failure threshold detection**
- Find the shadow fraction where each matcher drops below threshold
- Output: `max_shadow_fraction` for each matcher
- **Reveals:** At what shadow coverage does each matcher fail?

---

### Phase 4: Diagnostic Visualization

**Why:** What's solid in the original plan (unchanged).

**Task 4.1: Failure overlay images**
- For each failed pair, draw:
  - Green: correct matches
  - Red: incorrect matches  
  - Yellow: missed keypoints (detected in A but not in B)
- **Reveals:** WHERE on the image failures occur (shadows? textureless areas?)

**Task 4.2: Stage breakdown panel**
- For each pair, show 4 panels:
  1. Detection (keypoints overlaid)
  2. Description (descriptor distances)
  3. Matching (correct vs incorrect)
  4. Verification (warped vs reference)
- **Reveals:** WHICH stage failed first

**Task 4.3: Illumination heatmap**
- X-axis: shadow fraction
- Y-axis: matcher name
- Color: success rate
- **Reveals:** Quick comparison of all matchers across illumination levels

---

### Phase 5: Statistical Analysis

**Task 5.1: Per-matcher failure report**
```json
{
  "matcher": "sift-lightglue",
  "max_shadow_fraction": 0.35,
  "failure_mode": "descriptor_degradation",
  "detection_repeatability_at_failure": 0.45,
  "precision_at_failure": 0.32,
  "recommended_preprocessing": "CLAHE"
}
```

**Task 5.2: Correlation analysis**
- Correlate failure with:
  - Shadow fraction
  - Terrain relief
  - GSD (normalized AUC thresholds)
- **Reveals:** WHAT causes failure (shadows? terrain? resolution?)

**Task 5.3: GSD-normalized AUC**
- Compute AUC at thresholds: 1 GSD, 3 GSD, 5 GSD
- **Reveals:** End-to-end registration quality (normalized for resolution)

---

## Metrics Summary (Corrected)

| Stage | Metric | Range | What It Reveals |
|-------|--------|-------|-----------------|
| Detection | Repeatability | [0, 1] | Keypoint survival |
| Description | Descriptor distance (matcher-aware) | [0, ∞) | Vector stability |
| Matching | Precision, Recall | [0, 1] | Match correctness |
| Verification | H_error or reproj_error | [0, ∞) | Registration accuracy |
| **Overall** | **AUC @ 1/3/5 GSD** | [0, 1] | End-to-end quality |
| **Overall** | **MR @ 100P** | [0, 1] | Robustness at high precision |
| **Failure** | **Max shadow fraction** | [0, 1] | Illumination tolerance |

---

## Implementation Priority (Revised)

1. **Phase 0** (GT validation) — 2-3 hours
2. **Phase 1** (Dataset distribution analysis) — 1-2 hours
3. **Phase 2** (Stage metrics with corrected GT) — 3-4 hours
4. **Phase 3** (Illumination-stratified analysis) — 2-3 hours
5. **Phase 4** (Visualization) — 2-3 hours
6. **Phase 5** (Statistical analysis) — 1-2 hours

---

## Expected Output

For each matcher, you'll get a failure profile like:
```
sift-lightglue:
  Detection repeatability: 0.82 (nadir), 0.65 (oblique), 0.41 (dynamic)
  Max shadow fraction: 0.35
  Failure mode: descriptor_degradation (shadows change gradient direction)
  AUC @ 3 GSD: 0.91 (nadir), 0.72 (oblique), 0.45 (dynamic)
  MR @ 100P: 0.85 (nadir), 0.61 (oblique), 0.28 (dynamic)
  Terrain relief sensitivity: fails when relief > 0.2
```

---

## Open Questions

1. **Do we need to download the full dataset?** Our samples only have 3 images per split. The full dataset has 50,000+ pairs with depth maps. For meaningful statistics, we likely need the full dataset.

2. **Are solar angles stored anywhere?** The paper specifies them per lighting condition, but they're not in the `.npz` files. We may need to contact the authors or use the trajectory generation scripts.

3. **What's the correct threshold for "non-planar" terrain?** This depends on the application. For landing hazard detection, even small relief matters. For orbital matching, larger relief may be acceptable.
