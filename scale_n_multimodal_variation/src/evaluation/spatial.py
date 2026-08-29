import numpy as np


def match_distribution(
    keypoints,
    matches,
    image_shape,
    grid_rows=4,
    grid_cols=4
):
    """
    Measure how uniformly feature matches are distributed
    across an image grid.

    Returns
    -------
    coverage : float
        Fraction of grid cells containing at least one match.

    counts : numpy.ndarray
        Number of matches in each grid cell.

    uniformity : float
        Score in [0, 1]. Higher means more uniform distribution.
    """

    height, width = image_shape[:2]

    counts = np.zeros(
        (grid_rows, grid_cols),
        dtype=np.int32
    )

    if len(matches) == 0:
        return 0.0, counts, 0.0

    for match in matches:

        x, y = keypoints[match.queryIdx].pt

        col = int(x / width * grid_cols)
        row = int(y / height * grid_rows)

        col = min(col, grid_cols - 1)
        row = min(row, grid_rows - 1)

        counts[row, col] += 1

    occupied = np.count_nonzero(counts)

    coverage = occupied / counts.size

    mean = counts.mean()

    if mean == 0:
        uniformity = 0.0
    else:
        coefficient_of_variation = counts.std() / mean

        uniformity = 1.0 / (
            1.0 + coefficient_of_variation
        )

    return float(coverage), counts, float(uniformity)
