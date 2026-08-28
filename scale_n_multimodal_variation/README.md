# Pair 3 — Scale and Multimodal Registration

## Team Focus

Pair 3 investigates the scale and multimodal challenges in
SIH 2026 Problem Statement 26166.

### Person 5
Scale variation.

### Person 6
Multimodal correspondence, with primary focus on:

- IIRS ↔ LRO reference imagery
- IIRS ↔ TMC
- IIRS ↔ OHRC

## Objective

The first phase is to evaluate existing image-registration
methods on lunar imagery and identify their failure modes.

We are not developing the final novel solution during this phase.

## Initial Methods

- SIFT
- RIFT
- SuperPoint + LightGlue
- LoFTR

## IIRS Representations

- Selected spectral bands
- PCA components
- Gradient representation
- Local contrast
- Spatially aggregated representations

## Evaluation Metrics

- Number of matches
- Number of inliers
- Inlier ratio
- Reprojection RMSE
- Spatial coverage
- Runtime

## Experimental Question

How well can existing correspondence methods handle
the large sensor and spatial-resolution difference between
IIRS and panchromatic lunar imagery?

## Data

Scientific datasets are not stored in this repository.

See `data/README.md` for dataset information and product IDs.
