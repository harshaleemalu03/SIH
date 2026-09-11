# Multi-Modal, Sun-Angle and Scale-Invariant Lunar Image Correspondence

### Landscape, Gap Analysis, and a Minimal Baseline Plan
**Scope:** Aligning Chandrayaan-2 optical images (OHRC, TMC-2, IIRS) with reference lunar imagery
(LRO NAC, SELENE/Kaguya) at **sub-pixel accuracy** with **uniformly distributed** match points.

**Compiled:** August 2026. Research current to within this window.
**Evidence convention:** `[FACT]` = directly supported by a cited primary source;
`[INFERENCE]` = synthesis from evidence; `[REPORTED]` = community/practical account.

---

## 0. Executive Summary

The problem is genuinely hard because the three challenges (illumination / viewpoint / scale) do **not**
act independently — they compound. The literature shows a clear, consistent picture:

- **Illumination (sun-angle) is the dominant breaker.** Gradient-based descriptors (SIFT/SURF/ORB/KAZE)
  derive their signal from *shadow edges* on the Moon, and shadows physically **relocate** when the sun
  moves. SIFT's usable solar-azimuth tolerance before collapse is about **33°** in one lunar study
  `[FACT: Xie et al. 2025]`, and matches decay progressively across 20°–180° `[FACT: Wu et al. 2018]`.
- **Scale is the silent breaker.** Real cross-sensor GSD ratios are enormous (OHRC 0.25 m vs IIRS ~80 m
  ≈ **320×**; OHRC vs SELENE TC ≈ **40×**; OHRC vs NAC ≈ **2–8×**). No learning-based matcher is
  validated beyond ~6× scale ratio `[FACT: RUBIK, CVPR 2025]`, and the best of them still fails >45% of
  pairs even at that modest ratio.
- **Viewpoint is the *least* hard**, and the one classical methods (ASIFT) handle best — but ASIFT does
  nothing for illumination or large scale.
- **The cross-sensor lunar case is essentially unstudied end-to-end.** The only direct multi-algorithm,
  multi-sensor lunar benchmark is Makharia et al. 2025 (arXiv:2509.04775), and it shows learned SuperGlue
  + radiometric preprocessing currently wins, yet still depends on that preprocessing and fails on
  polar/SAR pairs. No published method delivers fully sun-angle- and scale-invariant correspondence
  across OHRC/TMC/IIRS ↔ LRO NAC / SELENE.

**Bottom line for the project:** A credible *baseline* is a classical, well-instrumented pipeline
(RootSIFT + geometric verification + sub-pixel refinement + uniformity control) that you will deliberately
**break** on a synthetic (rotation × scale × sun-angle) stress grid. The *innovation* target is a
cross-sensor, illumination-robust, scale-robust matcher — and this document maps exactly where today's
methods fail so you can position novel work.

---

## 1. Problem Framing and the Three Challenge Axes

### 1.1 Sensor fleet and the scale-ratio reality `[FACT]`

| Sensor | Mission | Band | GSD | Relevance to scale gap |
|---|---|---|---|---|
| **OHRC** | Chandrayaan-2 | Pan (450–800 nm) | **0.25 m** nadir @100 km | finest reference |
| **TMC-2** | Chandrayaan-2 | Pan (0.5–0.85 µm) | **5 m** @100 km | ~20× vs OHRC |
| **IIRS** | Chandrayaan-2 | Hyperspectral 0.8–5.0 µm (~250 bands) | **~80 m** @100 km | ~320× vs OHRC |
| **LRO NAC (L/R)** | LRO | Pan (mono) | **0.5–2.0 m/px** | ~2–8× vs OHRC |
| **SELENE TC** | Kaguya | Pan stereo | **10 m** | ~40× vs OHRC |
| **SELENE MI** | Kaguya | 9-band VIS/NIR | **20/60 m** | — |
| **LRO WAC** | LRO | 7-band | **~75–100 m** | ~400× vs OHRC |
| **LOLA / SLDEM** | LRO | Laser altimeter | 30–90 m | geodetic reference |
| **DFSAR / Mini-RF** | Chandrayaan-2 / LRO | SAR | 2–150 m | cross-*modality* |

**Cross-sensor ratio pairs you will face:**
OHRC:NAC ≈ 1:2–8 · OHRC:TMC-2 ≈ 1:20 · OHRC:SELENE-TC ≈ 1:40 · OHRC:IIRS ≈ 1:320 ·
OHRC:WAC ≈ 1:400 · IIRS:WAC ≈ 1:1.25 (closest matched pair).

> **Stress-test of the "scale-invariant" claim:** Most papers that say "scale-invariant" mean *within-frame
> affine scale* (a few ×, typically <2× training distribution). They do **not** mean 20–400× cross-sensor
> GSD ratios. Treat any bare "scale-invariant" claim with suspicion unless the tested ratio is stated.
> `[INFERENCE]` from RUBIK/MegaDepth evidence.

### 1.2 The three challenges (as they actually manifest on the Moon)

1. **Illumination variation** — Sun elevation (incidence) and azimuth change between acquisitions.
   On the Moon, near-homogeneous albedo means most *local texture is cast by shadows*. When the sun
   moves, **shadow edges relocate**, so both gradient orientation and intensity are not stably
   reproduced. `[FACT: Wu 2018; Pless-group WACV study — ~60% of matching decay occurs within the first
   5° of illumination-angle difference, dominated by *keypoint detection*, not description.]`
2. **Viewpoint variation** — Different camera positions/orientations. Measured by *transition tilt*
   (ASIFT). SIFT covers only 4 of 6 affine parameters. `[FACT: Yu & Morel 2009/2011.]`
3. **Scale variation** — Different sensor altitude/GSD producing the large ratios above. Standalone
   feature matchers tolerate only a few ×.

---

## 2. Landscape Map — Taxonomy of Approaches

```
                         LUNAR IMAGE CORRESPONDENCE
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────────┐
        │              │               │               │                  │
  CLASSICAL        ILLUMINATION-    STRUCTURE /      LEARNING-BASED      PHYSICS /
  FEATURE-BASED    INVARIANT        CRATER-BASED     MATCHERS            RADIOMETRY
  (SIFT, ASIFT,    (hist-adapt,     (CNSFM,          (SuperGlue,         (ISIS lronacpho,
   ORB, KAZE,      LIOP, RootSIFT,  shaded-DEM+      LoFTR, DKM, RoMa,   Hapke, SfS,
   Fourier–Mellin, Retinex/HE,      SIFT, crater     LightGlue,          photometric
   NCC/area)       phase-cong RIFT2) control nets)    MASt3R, MINIMA)     normalization)
        │              │               │               │                  │
   Breaks on      Survives moderate  Survives extreme  Best overall       Fixes the *cause*
   illum + large  illum but not      illum but fails  accuracy but        (sun-angle) but
   scale          polar/shadow       on sparse/       UNVALIDATED on      needs a DEM /
                    / cross-modal     uncratered       lunar + fails      model; NULLs at
                                     terrain           large scale        high incidence
```

### 2.1 Classical feature-based methods (SIFT family, ASIFT, Fourier–Mellin, NCC)

| Method | Illumination tolerance | Viewpoint (transition tilt) | Scale ratio | Typical accuracy | Documented failure |
|---|---|---|---|---|---|
| **SIFT** | collapses past ~**33°** azimuth diff `[Xie 2025]`; degrades 20°→180° `[Wu 2018]` | ~2 `[Yu&Morel]` | few×; ~5× w/ scale-restriction `[Sedaghat 2011]` | 0.2–2.5 px lit `[Makharia 2024]`; **173 px** mixed-light `[Stoken 2025]` | multi-illum + polar |
| **SURF** | fragile on uneven illum `[cea.ceaj 2020]`; only sim-proven | ~2 | few× | — | no lunar failure study (gap) |
| **ORB** | **worse** than SIFT mixed-light (**1091 px** RMSE) `[Stoken 2025]`; no true scale-inv | ~2 | poor across big GSD gap | lower SSIM than SIFT `[MoonMetaSync]` | OHRC↔TMC scale gap |
| **KAZE / AKAZE** | KAZE fails uneven illum `[cea.ceaj 2020]` | ~2.5 | few× | no sub-px on planetary `[EPSC2024]`; AKAZE fails polar + SAR-SELENE `[Makharia 2025]` | polar, SAR |
| **Harris / Harris-affine** | shadow-sensitive `[INF]` | 2.5 `[Yu&Morel]` | few× | most precise but most *unmatched* pairs `[EPSC2024]` | weak texture |
| **ASIFT** | **inherits SIFT illum fragility** | **32–36+** `[Yu&Morel]` | few× (no big-scale help) | ~1.0–1.3 px w/ IM-RANSAC `[Wang 2015]`; ~13.5–180× compute | polar, SAR `[Makharia 2025]` |
| **Fourier–Mellin / phase corr.** | moderate (global gain ok, **not** directional shadow) | rotation+scale ok | **≤1.8×** `[Reddy&Chatterji 1996]` | Template Match **17 px** mixed-light `[Stoken 2025]` | large scale + multimodal |
| **NCC / area / ASP** | fails large illum + low texture `[Rothenberger 2025]` | rigid only | any (intensity-based) | <5 m vs LOLA (ASP DTM) `[Moratto 2014]` | multi-illum pairs |

**Stress-test (each claim challenged):**
- *"SIFT is scale + rotation invariant."* True only for moderate zoom and rotation. At the lunar
  cross-sensor ratios it is **not** used as a scale-invariant matcher — pipelines instead use
  **camera-model/SPICE triangulation** (ASP/ISIS dense stereo), which is scale-agnostic *because it uses
  geometry, not feature scale*. `[FACT/INFERENCE]`
- *"ASIFT solves viewpoint."* Yes, but it does **nothing** for illumination and adds 13.5–180× compute;
  on polar/multi-illum lunar data Makharia 2025 report ASIFT *failed* alongside SIFT/AKAZE. `[FACT]`
- *"Fourier–Mellin handles rotation+scale."* Only up to **1.8×** scale on aerial images per the original
  authors; far below lunar ratios. `[FACT: Reddy & Chatterji 1996]`

### 2.2 Illumination-invariant correspondence methods

| Method | Mechanism | Reported gain | Documented failure |
|---|---|---|---|
| **Wu et al. 2018 hist-adapt SIFT** | levels SIFT orientation histogram to sub-solar azimuth; NCC matching | +40–60% matches on 20°–180° illum diff `[FACT]` | still weak at low sun / severe shadow (authors' own caveat) |
| **Zeng 2018 improved SIFT** | weights gradient by illumination orientation | ~40% more matches, works at 180° diff `[FACT]` | still gradient-based; untested at polar extremes |
| **CSIFT / PCA-SIFT / GSIFT** | colour/PCA/Leuven variants | small gains on *linear* brightness | CSIFT **does not help** under real illum `[MSR 2013]`; GSIFT wins only on *global* Leuven brightness, **not** sun-angle `[INF]`; IIRS/NAC are pan/grayscale → colour invariants irrelevant |
| **RootSIFT** | L1→sqrt of SIFT = Hellinger distance | big distinctiveness boost, zero storage | **descriptor-distance fix, not illumination fix** `[INF]` |
| **LIOP / UGSIFT** | intensity-rank order / unsigned gradients | good on *monotonic* (sqrt/log) changes | fails when shadow *edges relocate* (non-monotonic, geometric) `[SCIA 2013]` |
| **Retinex / HE / homomorphic + SIFT** | reflectance estimation preprocessing | better match counts in low light `[Lozano-Vázquez 2022; Li 2025]` | contrast/reflectance estimators, **not** BRDF inverters; can amplify noise on homogeneous regolith `[INF]` |
| **Phase-correlation illumination invariance (Wan 2014/15)** | decomposes cross-power spectrum; AD-SVD for azimuth reversal | ~0.05 px on **DEM-simulated** shading, azimuth up to 177° | validated on *synthetic* shading only (no albedo/cast-shadow/BRDF) `[INF]` |
| **RIFT / RIFT2 (phase congruency)** | edge-based, radiation-invariant descriptor | multimodal (IR/SAR/optical) where SIFT fails | **failed** OHRC–NAC polar & DFSAR–SELENE in Makharia 2025 `[FACT]`; cast shadows at low sun still breaker |
| **LIMA (Rothenberger 2025/26)** | correlation-based, built for extreme sun-angle (synthetic + real LRO NAC) | illumination-targeted classical method | area/correlation, not sparse descriptor; untested cross-sensor |

**Stress-test:**
- *"Illumination-invariant descriptors solve sun-angle."* They assume the **same local edges exist**
  under both illuminations. Lunar sun-angle change **relocates** shadow edges, so edge *location* changes.
  This is why even the best (Wu/Zeng) top out at +40–60% matches and still fail at low solar elevation.
  `[FACT + INFERENCE]`
- *"Leuven/HPatches illumination = lunar sun-angle."* **No.** Leuven is global linear brightness.
  Lunar sun-angle is non-linear, geometric shadow restructuring. A method winning on Leuven tells you
  nothing about the lunar regime. `[INFERENCE]`

### 2.3 Structure / crater-based methods (the strongest lunar-specific illumination answer)

| Method | Mechanism | Reported gain | Documented failure |
|---|---|---|---|
| **CNSFM — Crater Neighborhood Structure (Xie 2025)** | DL craters → one crater + K-nearest → geometric/topological match (not grayscale) | **100%** equatorial/mid-lat; **72.3%** polar; azimuth tol **147.7°** vs SIFT 33° `[FACT]` | fails on small overlap, heavy shadow, **sparse-crater** terrain (young units, melt ponds) |
| **Shaded-DEM + SIFT (IEEE 11185283; ISPRS 2026)** | render shaded DEM at *same* acquisition sun → match similarly-lit renders | **0.83 px RMS** south-pole network `[FACT]` | needs a LOLA DEM; not direct image-to-image |
| **Crater control network (ISPRS 2026 XLIX-B2-467)** | YOLOv8 craters + virtual feature points | 1.8M–2.2M tie points, RMS 0.83 px | depends on crater detection quality |

**Stress-test:** CNSFM is the best published lunar multi-illumination method, but it **drops to 72.3% at
poles** and fails on uncratered/permanently-shadowed overlap. It is domain-specific (needs lunar crater
training) and does **not** generalize to non-lunar bodies without retraining. `[FACT + INFERENCE]`

### 2.4 Learning-based matchers (2021–2026)

| Method | Class | Best at | Documented weakness for *lunar cross-sensor* |
|---|---|---|---|
| **LoFTR / QuadTree / MatchFormer / TopicFM** | detector-free transformer | illumination + small viewpoint (HPatches ILLUM 0.98) | **deteriorates completely under large rotation** `[Bökman&Kahl 2022]`; trained on MegaDepth <2× scale; no cross-sensor test |
| **SE2-LoFTR / RA-LoFTR** | rotation-equivariant | in-plane rotation | gains modest; no scale/cross-sensor test |
| **SuperGlue / LightGlue / SGMNet / OAMatcher** | graph/sparse | adaptive speed (LightGlue ~10× faster) | keypoint-dependent; SuperPoint repeatability on low-texture lunar terrain untested; MegaDepth/ScanNet-trained |
| **DKM / RoMa / RoMa v2** | dense, frozen DINOv2 | WxBS +36% mAA; best OOD | **DKM explicitly resolution-sensitive** `[README]`; RoMa still struggles extreme viewpoint (RUBIK); modality (IR↔RGB) weakest subset; **SatAst only 37 AUC@10px** (astronaut↔satellite = closest lunar proxy) `[FACT]` |
| **DUSt3R / MASt3R** | 3D foundation model | viewpoint up to 180° (MapFree) | RGB-native; computational; no lunar validation; grayscale tiling untested |
| **MINIMA (CVPR 2025)** | cross-modal data-engine | fine-tunes any matcher on RGB↔IR↔SAR↔Depth↔Event; zero-shot Optical-SAR | **most portable recipe** but validated on Earth/SAR/IR/medical, **not lunar reflectance-under-different-sun**; synthetic modality transfer |

**Stress-test (the crucial one):**
- *"Learning-based matchers are SOTA, so use them."* True on **terrestrial** benchmarks. But:
  1. **Training-distribution gap** — almost all are trained on MegaDepth (internet photos, <2× scale,
     similar illumination). None validated on lunar multi-sensor data. `[FACT]`
  2. **Scale-ratio gap** — RUBIK (CVPR 2025) is the largest structured scale test (≤6×); even the best
     method succeeds on only **54.8%** of pairs there, with failures concentrated at low-overlap ×
     large-scale × extreme-viewpoint. Lunar 20–400× ratios are an order of magnitude beyond the top
     tested bin. `[FACT → INFERENCE]`
  3. **Modality ≠ sun-angle** — "cross-modal" methods handle RGB↔IR↔SAR appearance gaps, **not** the
     same-modality-but-different-illumination-*geometry* case (both images grayscale reflectance, but
     shading geometry flips). This regime is poorly covered. `[INFERENCE]`
  4. **DKM is explicitly resolution-fragile** — feeding different resolutions "may give significantly
     worse performance" (authors' README). Directly relevant to cross-GSD matching. `[FACT]`

### 2.5 Physics / radiometry-based normalization (attacks the *cause*, not the symptom)

- **ISIS `lronacpho`** — empirical photometric normalization to reference incidence/emission/phase.
  Valid only for **phase 15°–65°**; pixels with **incidence >90° set to NULL** (i.e., fails exactly where
  matching is hardest — poles). `[FACT: ISIS doc]`
- **Hapke / Boyd global solution** — physical BRDF. Boyd 2014 normalizes NAC mosaics well in
  shadow-free regions, but **"shadows are the largest contributor to noise"** in normalized NACs. `[FACT]`
  Shepard & Helfenstein 2007 show Hapke inversion parameters are **not uniquely interpretable** as
  physical properties (retrieved albedo can err 2×). Grumpe & Wöhler 2012: Hapke normalization **does not
  eliminate** all topography-related spectral distortion. `[FACT]`
- **ASP Shape-from-Shading (SfS)** — Lunar-Lambertian refinement; used to align OHRC DEM vertically to
  NAC DTM. `[FACT: Aadi 2026]`

**Stress-test:** Photometric normalization assumes a *single homogeneous BRDF per tile*. Lunar regolith is
heterogeneous (mare:highland ratio shifts with incidence; young craters/melt anomalous), and the roughness
parameter is unconstrained at high incidence — exactly the polar regime. Permanent shadow → no signal to
normalize. `[FACT + INFERENCE]`

### 2.6 Open-source software / pipelines

| Tool | Approach | Lunar readiness |
|---|---|---|
| **USGS ISIS3** | `jigsaw` bundle adjustment; `autoseed`/`pointreg`/`findfeatures` control nets; `lronacpho` photometric | Primary ingest/camera model for LRO NAC, Kaguya TC, **now Chandrayaan-2 via PDS4+CSM** |
| **NASA Ames Stereo Pipeline (ASP)** | `parallel_stereo` (NCC/SGM), `bundle_adjust` (Ceres), `pc_align` (ICP to DTM), SfS | De-facto lunar stereo engine; NAC/OHRC/TMC/TC DEMs |
| **AutoCNet** | Python sparse CV correspondence + bundle adj | Planetary control nets (Apollo 15 demo); not Chandrayaan-specific |
| **AROSICS / AROP / OSSIM / MICMAC / COSI-Corr** | phase-corr / area / generic SfM / orthorectification | General-purpose; need lunar sensor-model wrappers; **no turnkey Chandrayaan-2→LRO co-registration exists** |

> **Stress-test:** There is **no published rigorous cross-sensor bundle-adjustment network** that jointly
> solves Chandrayaan-2 and LRO camera models in one adjustment. Cross-sensor tying is done *indirectly*
> via ICP-to-NAC-DTM or via WAC/SELENE as intermediate control. `[INFERENCE]` This is the central open
> problem.

---

## 3. Existing Work Directly on Chandrayaan-2 / LRO / SELENE

- **OHRC ↔ LRO NAC 2D registration** — Makharia et al. 2024 (InGARSS, doi:10.1109/ingarss61818.2024.10984368):
  SIFT/ASIFT, **0.2–2.5 px** accuracy (equatorial, similar illumination). `[FACT]`
- **OHRC ↔ NAC + multi-sensor benchmark** — Makharia et al. 2025 (arXiv:2509.04775): SIFT, ASIFT, RIFT2,
  AKAZE, **SuperGlue** on OHRC–NAC, IIRS–WAC, DFSAR–SELENE (equatorial + polar). **SuperGlue best**
  (lowest RMSE, fastest); classical methods degrade under polar lighting; RIFT2 failed polar/SAR pairs.
  **This is the only direct multi-sensor lunar evaluation.** `[FACT]`
- **OHRC DEM ↔ NAC DTM** — Aadi et al. 2026 (arXiv:2604.01032): open-source ASP/ISIS/CSM; ICP to NAC DTM,
  vertical RMSE **5.85 m**, horizontal ~30 cm. Tungathurthi 2026 (arXiv:2602.14993): 0.30 m Vikram-site
  DEM, CSM camera models mandatory. `[FACT]`
- **TMC-2 DEM** — LPSC2020 #1127: control points from **SELENE ortho + LOLA**, not LRO. `[FACT]`
- **IIRS ↔ WAC** — preprint 2025: SIFT auto-selenoreferencing, RMS << one IIRS pixel. `[FACT]`
- **Optical ↔ SAR** — Xu et al. 2025 (Remote Sensing 17(4):613): Mini-RF/Mini-SAR to DEM via simulated
  SAR + NCC; error ~3000 m → ~100 m. `[FACT]`

**Stress-test of "it's been solved":** Every Chandrayaan-2↔LRO/SELENE geometric result is either (a) 2D
homography/feature-based for *image overlay* (fails polar/cross-modal), or (b) 3D ICP of *DEMs*. There is
**no** end-to-end cross-sensor matching validated across the full sun-angle range with full illumination
invariance. `[FACT + INFERENCE]`

---

## 4. Evaluation Metrics & Benchmarks (for your baseline)

### 4.1 Metric stack to adopt (transferable from terrestrial benchmarks)
- **Sub-pixel accuracy:** RMSE of tie/check points in **px and m**; report **median** and **max** (robust
  to outliers). `[FACT: Stoken 2025 reports 1.75 px / 1.36 m; ISPRS 2026 reports 0.83 px RMS.]`
- **Correspondence quality:** inlier **count**, inlier **ratio**, precision/recall/F1.
- **Downstream:** homography **AUC @3/5/10 px** (HPatches-style); **mAA** (IMC; *always state the variant
  and how unregistered images are handled* — mAA is ambiguous otherwise `[FACT: arXiv:2408.16445]`);
  reprojection error (4-corner).
- **Uniform distribution (your explicit requirement):** **VMR** (Variance-to-Mean Ratio; target <1 =
  uniform) `[Zhang 2020]`; **Delaunay Qₚ** `[Mahmood&Lee 2019]`; **Voronoi α**; **coverage %**; tie-point
  multiplicity.

### 4.2 Benchmarks — and their gaps
- **HPatches** (illumination + viewpoint homography) — canonical, but **planar, short-baseline, synthetic
  warps**; illumination split is harder than viewpoint `[FACT: Balntas 2017]`. *Not lunar.*
- **RUBIK (CVPR 2025)** — the cleanest scale-robust protocol (bins overlap × scale-ratio × viewpoint).
  Best method 54.8% at ≤6×. *Directly relevant protocol to imitate.*
- **IMC2023/2024** — phototourism/urban, mAA metric. *Metric transferable, data not.*
- **WxBS / SatAst** — closest OOD analogs (multi-sensor, astronaut↔satellite). *Still not lunar.*
- **No public lunar correspondence benchmark exists.** Ground truth is manual tie points, LOLA, or ISIS
  control networks. `[FACT]` → **You must manufacture synthetic transforms for stress-testing.**

### 4.3 Designing the baseline stress grid `[INFERENCE, standard practice]`
Generate ground-truth homographies and report a **success-rate heatmap** over:
- **Rotation** θ ∈ {0°, 15°, …, 345°}
- **Scale** s ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0}× (push to your real ratios)
- **Illumination / sun-angle** — gamma/brightness, *and* **synthetic hillshade from a DEM** with varied
  azimuth/elevation (this is the honest sun-angle proxy), plus histogram matching and noise.
- **Overlap** ∈ {30%, 50%, 70%} (reject border-collapse cases).
Report RMSE(px/m), inlier ratio, homography AUC, VMR, and failure rate per cell.

> **Stress-test of metrics themselves:** SSIM/MSE/PSNR are **unreliable under large illumination change**
> (susceptible to local minima, inconsistent) `[FACT: SiamRegQC, Kulkarni 2024]`. Use them only as
> secondary; lead with RMSE / inlier ratio / AUC. mAA must ship with its exact variant.

---

## 5. Minimal Baseline Implementation Plan (High-Level, No Code)

Goal: a **working, instrumented baseline** that establishes *what today's best practical pipeline can and
cannot do* on your exact data, before any research. Use the **latest practical algorithms**, not toy code.

### Stage A — Data acquisition & preparation
1. Pull co-located pairs from **PRADAN** (OHRC/TMC/IIRS + SPICE) and **LROC QuickMap/PDS** (NAC),
   **SELENE DARTS** (TC/MI). Derive OHRC stereo pairs from metadata (not pre-distributed) `[FACT: Aadi 2026]`.
2. Build a **small curated validation set** with manually-tied control points (expert tie) and/or LOLA-
   anchored reference — this is your only real lunar ground truth.
3. For each pair, record **metadata-driven challenge labels**: incidence/azimuth difference, GSD ratio,
   overlap, latitude (equator vs polar).

### Stage B — Classical baseline pipeline (the "honest floor")
1. **Feature extraction/description:** SIFT → **RootSIFT**; run **ASIFT** as the viewpoint-robust variant
   and **AKAZE/ORB** as scale/rotation contrasts. Compare against **SuperGlue / LightGlue** (learned
   sparse) and **LoFTR / RoMa** (learned dense) as stronger references.
2. **Radiometric preprocessing (the illumination lever):** CLAHE / Retinex, and **ISIS `lronacpho`**
   photometric normalization where phase 15°–65°. Test *with vs without* to isolate its contribution.
3. **Matching:** symmetric nearest-neighbour + SNN ratio test; keep intersection.
4. **Geometric verification:** RANSAC / MAGSAC++ / Degensac estimating **homography** (local tiles) or
   **affine** (OHRC vs NAC local), threshold 0.5–1 px.
5. **Warp + sub-pixel refinement:** homography/affine warp; refine with **optical-flow / ELA / mutual-
   information-NCC** local deformation.
6. **Uniformity control:** stratified/bucketed inlier selection (SRS-RANSAC) to spread points; report VMR
   and coverage.

### Stage C — The stress grid (the core experiment)
Run Stage B across the synthetic **(rotation × scale × sun-angle × overlap)** grid (§4.3) on a
representative texture patch, **and** on the real curated lunar pairs. Produce the success-rate heatmap.
This is what tells you *where* the baseline breaks and *quantifies* each challenge's cost.

### Stage D — Reporting (what "done" looks like)
For every configuration, log: #putative, #inliers, inlier ratio, **RMSE(px & m)**, median/max error,
homography AUC@3/5/10 px, **VMR / coverage %**, runtime, and failure rate. Compare classical vs learned,
with vs without photometric normalization, per challenge axis.

### Why this is the right minimal plan
- It uses the **latest practical algorithms** (RootSIFT, LightGlue/SuperGlue, RoMa, MAGSAC++, MINIMA-style
  thinking) without requiring you to invent anything yet.
- It produces the **exact metrics** the problem asks for (sub-pixel RMSE, inlier count/ratio, uniform
  distribution).
- It **isolates each challenge** via the stress grid, so your later innovation has a quantified baseline
  to beat and a clear failure map to target.

---

## 6. Where the Innovation Opportunity Actually Is

Synthesizing all of the above, the **unfilled gaps** (each a candidate research contribution):

1. **Cross-sensor scale robustness (5–400×).** No method validated beyond ~6×. Needed: a scale-equivariant
   / multi-resolution frontend (ASIFT-style scale-space + transformer attention) that covers real GSD
   ratios. `[INFERENCE]`
2. **True sun-angle invariance under relocated shadows.** Gradient/phase descriptors fail when shadow
   *edges move*. Structure/crater methods work but fail on sparse/uncratered terrain. A learned,
   shadow-geometry-aware or self-supervised (BRDF-relit) descriptor is open. `[INFERENCE]`
3. **End-to-end cross-sensor lunar matching** (OHRC/TMC/IIRS ↔ NAC/SELENE) with no indirect ICP/control
   intermediary. `[INFERENCE]`
4. **A lunar correspondence benchmark** binned by GSD ratio × incidence/azimuth difference — because no
   public one exists and it is required to claim SOTA. `[FACT/INFERENCE]`
5. **MINIMA-style synthetic cross-sensor data engine** built from co-registered OHRC+TMC+IIRS+NAC+SELENE —
   the most portable recipe to adapt learned matchers to your domain. `[INFERENCE, based on Ren et al. 2025]`

---

## 7. Caveats on This Document

- Claims tagged `[FACT]` are drawn from the cited primary sources (papers, NASA/USGS/JAXA docs, benchmark
  pages). Several *illumination-tolerance* numbers (SIFT 33°, CNSFM 147.7°) are **dataset- and threshold-
  dependent** success-rate cutoffs — treat them as *relative ordering*, not absolute physical limits.
- Several classical-method (SURF, Harris) lunar failure modes are `[INFERENCE]` from gradient-descriptor
  theory and Earth/HRSC results, because no dedicated lunar study was found (a gap itself).
- "nasalign" is a misnomer sometimes seen in discussion; the real ASP tools are `pc_align` / `n_align`,
  and ISIS uses `findfeatures`/`pointreg`/`jigsaw`. `[INFERENCE/correction]`

---

## 8. Key References (primary sources, with URLs)

**Classical / planetary:**
- Wu, Zeng, Hu 2018, *Planetary & Space Science* 152:45–54 — https://doi.org/10.1016/j.pss.2018.01.007
- Yu & Morel 2009/2011, *ASIFT*, SIAM J. Imaging Sci. 2(2):438–469 — https://doi.org/10.1137/080732730 ; IPOL https://www.ipol.im/pub/art/2011/my-asift/
- Xie et al. 2025, *CNSFM*, Remote Sensing 17(13):2302 — https://doi.org/10.3390/rs17132302
- Stoken et al. 2025, *Automated Coregistration of LROC NAC*, LPSC #2602 — https://www.hou.usra.edu/meetings/lpsc2025/pdf/2602.pdf
- Reddy & Chatterji 1996, *Fourier–Mellin*, IEEE TIP 5(8).
- Moratto et al. 2014, *ASP LRO-NAC DTMs*, LPSC #2892 — https://www.hou.usra.edu/meetings/lpsc2014/pdf/2892.pdf
- Edmundson et al. 2012, *JIGSAW*, ISPRS Ann. I-4:203 — https://doi.org/10.5194/isprsannals-i-4-203-2012
- Makharia et al. 2024, InGARSS — https://doi.org/10.1109/ingarss61818.2024.10984368
- Makharia et al. 2025, *Chandrayaan-2 cross-sensor eval*, arXiv:2509.04775 — https://doi.org/10.48550/arxiv.2509.04775
- Aadi et al. 2026, *OHRC multi-view DEM*, arXiv:2604.01032 ; Tungathurthi 2026, arXiv:2602.14993
- Rothenberger et al. 2025/26, *LIMA*, AIAA 2025-2073 / 2026-2244
- IEEE 11185283 (shaded-DEM+SIFT); ISPRS 2026 XLIX-B2-467 (crater control net)
- Wan et al. 2014/15, *Phase Correlation illum.*, doi:10.1109/tgrs.2015.2429740
- Li et al. 2023, *RIFT2*, arXiv:2303.00319

**Learning-based:**
- LoFTR — https://arxiv.org/abs/2104.00680 ; QuadTree — https://arxiv.org/abs/2201.02767
- SuperGlue (CVPR 2020); LightGlue — https://arxiv.org/abs/2306.13643
- DKM — https://arxiv.org/abs/2202.00667 ; RoMa — https://arxiv.org/abs/2305.15404 ; RoMa v2 — https://arxiv.org/abs/2511.15706
- DUSt3R — https://arxiv.org/abs/2312.14132 ; MASt3R — https://arxiv.org/abs/2406.09756
- MINIMA (CVPR 2025) — https://arxiv.org/abs/2412.19412
- RUBIK (CVPR 2025) — https://arxiv.org/abs/2502.19955 ; GIM/ZEB — https://arxiv.org/abs/2402.11095

**Metrics / benchmarks:**
- HPatches — https://hpatches.github.io/ (Balntas et al. CVPR 2017)
- IMC — https://image-matching-challenge.github.io/ ; Hexathlon arXiv:2407.03172
- mAA critique — arXiv:2408.16445 ; BIRL — arXiv:1912.13452
- Spatial uniformity: Zhang 2020 https://doi.org/10.1080/22797254.2020.1724519 ; Mahmood&Lee 2019 doi:10.1109/access.2019.2936647
- SiamRegQC (metric caveat) — https://proceedings.mlr.press/v250/kulkarni24b.html

**Software / data:**
- ASP — https://stereopipeline.readthedocs.io/en/latest/examples/lronac.html
- ISIS `lronacpho` — https://isis.astrogeology.usgs.gov/8.3.0/Application/.../lronacpho.html
- AutoCNet — https://github.com/DOI-USGS/Autocnet ; AROSICS — https://github.com/GFZ/arosics
- PRADAN (Chandrayaan-2) — https://pradan.issdc.gov.in/ch2/ ; LROC QuickMap — https://quickmap.lroc.im-ldi.com ; SELENE DARTS — https://darts.isas.jaxa.jaxa.jp/planet/pdap/selene

---

## 9. Illumination-Only Focus (current sprint): Essential Reading, Datasets & Baseline Sketch

**Scope lock:** This section isolates the **sun-angle / illumination axis only**. Scale and viewpoint are
held fixed (same GSD-ish, near-identical geometry, different sun) so any accuracy drop is attributable to
illumination and nothing else. `[INFERENCE, standard controlled-experiment practice]`

### 9.1 Must-read papers (illumination-specific) — filtered from the full landscape above

**Tier 1 — read first**
- **Xie 2025 — CNSFM** (best published lunar multi-illumination method; 100% equator / 72.3% polar; azimuth
  tol **147.7°** vs SIFT 33°). The strongest illumination-specific *reference method* to beat.
  → https://doi.org/10.3390/rs17132302
- **Makharia 2025 — SuperGlue on Chandrayaan-2** (your proven SOTA baseline on real Chandrayaan data).
  → https://doi.org/10.48550/arxiv.2509.04775
- **Wu 2018 — illumination-invariant SIFT** (histogram adaptation to sub-solar azimuth; +40–60% matches on
  20°–180° illum diff; the foundational classical fix).
  → https://doi.org/10.1016/j.pss.2018.01.007
- **LomFormer 2026** (Transformer that *renders* an image under a different sun angle from DEM; attacks
  illumination as a controllable variable — directly relevant to your framing).
  → https://doi.org/10.7527/S1000-6893.2025.32874
- **Stoken 2025 — LROC NAC coregistration** (deep matcher 1.75 px vs SIFT 173 px under mixed lighting; shows
  the failure mode you are solving). → https://www.hou.usra.edu/meetings/lpsc2025/pdf/2602.pdf

**Tier 2 — methods to know**
- **RIFT2** (phase-congruency, radiation-invariant descriptor; multimodal). → https://arxiv.org/abs/2303.00319
- **Wan 2014/15** (phase-correlation illumination invariance). → https://doi.org/10.1109/tgrs.2015.2429740
- **Pless-group WACV study** (why matching collapses in first 5° of illumination change — keypoint detection,
  not description). → https://www2.seas.gwu.edu/~pless/papers/wacvFeaturesOverTime.pdf
- **MoonMetaSync** (empirical proof low sun-angle wrecks SIFT on Chandrayaan TMC-2↔OHRC).
  → https://arxiv.org/abs/2410.11118
- **ISIS `lronacpho` doc** (operational photometric normalization — the "cause-level" fix; valid 15–65°
  phase, NULLs >90° incidence). → https://isis.astrogeology.usgs.gov/8.3.0/Application/presentation/Tabbed/lronacpho/lronacpho.html
- **HPatches** (benchmark *protocol* — illumination split is the harder split; use methodology, not data).
  → https://hpatches.github.io/

**Deliberately excluded for this sprint** (not illumination-specific): Tianwen-1/Mars, Geo-LoFTR/Mars,
WARG/LunarLoc (localization), Aadi/Tungathurthi (DEM photogrammetry), ISPRS 2026 crater control network
(viewpoint/distribution).

### 9.2 Datasets you can actually work with (illumination-only)

| Dataset | Real/Synth | Public? | Illumination GT? | Good for illum-only? | Link |
|---|---|---|---|---|---|
| **MoonAnything (LunarPhoto)** | Real LRO + SVBRDF render | **Yes** (GitHub) | Multi-illum + reflectance | **Best** — same geometry, different sun, dense supervision | https://github.com/clementinegrethen/MoonAnything |
| **StereoLunar** | Synthetic (Hapke BRDF) | **Yes** (Zenodo) | Depth + sun angles | **Yes** — pick same-trajectory/different-sun subsets | https://zenodo.org/records/17514376 |
| **LunarG2R** | Real LRO + Tycho DEM | **Yes** (Zenodo) | Geometry↔appearance, multi-light | Yes — illumination/reflectance learning | https://zenodo.org/records/17964776 |
| **CRESENT-365** | Synthetic | **Yes** (via STELLA) | Year-long sun-angle cycle | Yes — extreme illumination coverage | https://doi.org/10.1007/s42064-025-0294-0 |
| **LROC NAC repeat obs.** | **Real** | Yes (PDS/QuickMap) | Known poses; no dense corr. GT | **Best real validation** — build multi-illum pairs | https://quickmap.lroc.im-ldi.com |
| **Chandrayaan-2 OHRC (PRADAN)** | **Real** | Yes (ISRO) | Limited poses | **Your target domain** — pairs at diff. sun angles | https://pradan.issdc.gov.in/ch2/ |
| **MiLOIs** (Xie 2025) | Real LROC NAC | Unclear (verify) | 321 multi-illum pairs | Reference protocol if released | in Xie 2025 |
| **HSROSS** (Luo 2024) | Real satellites | Yes | Radiometric-difference scenarios | Optional proxy (not lunar) | https://arxiv.org/abs/2405.06246 |

**Stress-test caveat:** synthetic datasets use the **Hapke BRDF**, a simplification of real regolith
reflectance. Models trained *only* on them may under-perform on real non-Lambertian lunar terrain → real
LRO/Chandrayaan validation is non-negotiable, not optional. `[FACT + INFERENCE]`

### 9.3 Baseline experiment sketch (illumination-only, using §9.2)

**Train / evaluate on (public, sun-angle supervision, geometry held fixed):**
- **MoonAnything `LunarPhoto`** — same geometry, different sun, dense GT. Primary illum-controlled set.
- **StereoLunar** — select subsets with *identical trajectory but different sun angle* to isolate illum.

**Validate on real data (no dense GT; use known poses / manual tie points):**
- **LROC NAC** multi-illumination pairs you build from QuickMap (select repeat observations at different
  incidence/azimuth).
- **Chandrayaan-2 OHRC** pairs at different sun angles from PRADAN (the end goal).

**Compare against:** SuperGlue (Makharia) + classical illumination fixes (Wu 2018 SIFT, CNSFM, RIFT2) +
photometric normalization (`lronacpho`). **Hold scale and viewpoint constant** so the only ablation variable
is illumination; report RMSE(px & m), inlier ratio, homography AUC@3/5/10 px, median/max error per
incidence/azimuth-difference bin (mimic the §4.3 grid but sun-angle only).

**Success criterion for the sprint:** quantify each method's usable azimuth/incidence tolerance and identify
the illumination regime (low sun / high azimuth swing) where today's best practical pipeline breaks — that
boundary is your innovation target.
