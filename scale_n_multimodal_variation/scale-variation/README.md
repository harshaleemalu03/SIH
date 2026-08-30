# Scale Variation Analysis for Lunar Image Matching

## Overview

This project evaluates the robustness of modern image matching algorithms under scale variations in lunar surface imagery.

The objective is to determine how well different feature matching techniques perform when the same image is observed at different scales, which is a common challenge in satellite and planetary image analysis.

---

## Problem Statement

Feature matching is an important step in:

- Image registration
- Planetary mapping
- Localization and navigation
- Change detection
- Remote sensing applications

However, images of the same region may appear at different scales due to:

- Different sensor resolutions
- Varying altitudes
- Zoom levels
- Cropping operations

This experiment evaluates the impact of scale variation on matching performance.

---

## Algorithms Evaluated

The following state-of-the-art matchers were tested:

1. LoFTR
2. SuperPoint + LightGlue
3. SIFT + LightGlue

---

## Dataset

The experiments were performed using lunar surface images obtained from:

- OHRC (Orbiter High Resolution Camera)
- TMC (Terrain Mapping Camera)

Images were manually selected and resized to generate multiple scale variants.

---

## Scale Variations Tested

The following scale factors were used:

| Scale |
|---------|
| 0.25x |
| 0.50x |
| 0.75x |
| 1.00x |
| 1.25x |
| 1.50x |
| 2.00x |
| 3.00x |
| 4.00x |
| 6.00x |

---

## Evaluation Metrics

For each algorithm and scale factor, the following metrics were recorded:

- Number of matches
- Number of inliers
- Inlier ratio

Where:

Inlier Ratio = Inliers / Matches

Higher inlier ratios indicate more reliable matches.

---

## Methodology

1. Load the original image.
2. Generate scaled versions of the image.
3. Run each matching algorithm.
4. Compute:
   - Matches
   - Inliers
   - Inlier Ratio
5. Store results in CSV files.
6. Visualize trends across scales.

---

## Running the Experiment

Open the notebook in Google Colab.

Install dependencies:

```python
!git clone --recursive https://github.com/gmberton/vismatch.git
%cd vismatch
!pip install -r requirements.txt
!pip install -e .
```

Upload the input image and execute all cells.

The notebook automatically:

- Creates scaled image variants
- Executes matching algorithms
- Collects metrics
- Produces CSV outputs and plots

---

## Applications

- Planetary image registration
- Lunar mapping
- Remote sensing
- Satellite image alignment
- Visual localization
- Feature matching research
