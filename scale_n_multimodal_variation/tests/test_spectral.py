import numpy as np

from src.preprocessing.spectral import compute_pca


def test_pca_shape():

    cube = np.random.rand(
        20,
        30,
        10
    ).astype(np.float32)

    components, model = compute_pca(
        cube,
        n_components=3
    )

    assert components.shape == (20, 30, 3)
    assert len(model.explained_variance_ratio_) == 3
