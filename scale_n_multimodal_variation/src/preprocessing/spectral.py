import numpy as np
from sklearn.decomposition import PCA


def compute_pca(cube, n_components=3):
    """
    Compute PCA on an IIRS hyperspectral cube.

    Parameters
    ----------
    cube : numpy.ndarray
        Hyperspectral cube with shape (height, width, bands).

    n_components : int
        Number of principal components.

    Returns
    -------
    components : numpy.ndarray
        PCA components with shape
        (height, width, n_components).

    model : PCA
        Fitted PCA model.
    """

    height, width, bands = cube.shape

    pixels = cube.reshape(-1, bands)

    valid = np.all(np.isfinite(pixels), axis=1)

    model = PCA(
        n_components=n_components
    )

    transformed = np.zeros(
        (pixels.shape[0], n_components),
        dtype=np.float32
    )

    transformed[valid] = model.fit_transform(
        pixels[valid]
    )

    components = transformed.reshape(
        height,
        width,
        n_components
    )

    return components, model
