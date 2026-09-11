# Illumination Variation Robust Registration

This module provides tools, models, and benchmarking pipelines for image registration under extreme illumination variations on lunar surface imagery (LROC / Chandrayaan / StereoLunar).

---

## 🌟 Features

- **Extreme Illumination Invariance**: Robust feature matching and dense correspondence under severe shadow, phase angle, and lighting discrepancies.
- **Deep Matchers & Baselines**:
  - **Fine-Tuned EfficientLoFTR**: Custom lunar-domain fine-tuned checkpoint for linear-attention transformer dense matching.
  - **SuperPoint + LightGlue / SuperGlue**: Sparse keypoint extraction and graph neural matching.
  - **RoMA & LoFTR**: Dense matching baselines.
  - **SIFT**: Classical baseline with robust RANSAC filtering.
- **Unified Registration & Evaluation Pipeline**:
  - Rigorous geometry conversion from orbit cam2world poses and depth maps.
  - Standardized metrics: Inlier Ratio, Epipolar AUC (@5°, @10°, @20°), Pose Error, Reprojection Error.
- **Interactive Web App**:
  - Streamlit/Flask UI (`app.py` & `static/`) for side-by-side registration visualization, interactive checkerboard comparison, and keypoint matching inspection.

---

## 📂 Directory Structure

```
illumination_variation/
├── app.py                      # FastAPI web demo application
├── static/                     # Web app frontend (HTML / CSS / JS)
├── benchmark.py                # Standardized registration benchmarking CLI
├── register_pair.py            # Pair registration script
├── process_lroc.sh             # LROC ISIS preprocessing script
├── run.sh                      # Launcher script
├── pyproject.toml              # Project dependencies & packaging
├── requirements.txt            # Python requirements
├── docs/                       # Benchmark plans and technical architecture
└── finetune/                   # Fine-tuning framework
    ├── docs/                   # Implementation blueprints & evaluation documentation
    ├── registration/           # Unified registration library (adapters, metrics, RANSAC, GT)
    ├── scripts/                # Benchmark, evaluation, and fine-tuning launchers
    └── EfficientLoFTR/         # EfficientLoFTR model architecture adapted for lunar imagery
```

---

## 🚀 Getting Started

### 1. Installation

Using `uv` (recommended) or `pip`:
```bash
cd illumination_variation
uv sync
# or: pip install -r requirements.txt
```

### 2. Launching the Interactive Web UI

```bash
python app.py
```
Open `http://localhost:5000` in your browser to inspect pairwise alignment, match vectors, and checkerboard blend views.

### 3. Running Pair Registration

```bash
python register_pair.py --source <path_to_img1> --reference <path_to_img2>
```

### 4. Running Benchmarks

```bash
python benchmark.py
# or for full evaluation across all models:
python finetune/scripts/benchmark_all_models.py
```

---

## 📊 Evaluation & Fine-Tuning

Detailed fine-tuning blueprints, geometry conversion details, and StereoLunar benchmark results are documented in:
- [`docs/ILLUMINATION_BENCHMARK_PLAN.md`](file:///home/ojas/projects/SIH/illumination_variation/docs/ILLUMINATION_BENCHMARK_PLAN.md)
- [`finetune/docs/FINETUNING_PLAN.md`](file:///home/ojas/projects/SIH/illumination_variation/finetune/docs/FINETUNING_PLAN.md)
- [`finetune/docs/EVALUATION.md`](file:///home/ojas/projects/SIH/illumination_variation/finetune/docs/EVALUATION.md)
