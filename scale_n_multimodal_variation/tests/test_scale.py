import numpy as np

from src.preprocessing.scale import (
    resize_image,
    scale_pyramid
)


def test_resize_image():

    image = np.zeros(
        (100, 200),
        dtype=np.uint8
    )

    result = resize_image(
        image,
        0.5
    )

    assert result.shape == (50, 100)


def test_scale_pyramid():

    image = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    pyramid = scale_pyramid(
        image,
        scales=(1.0, 0.5, 0.25)
    )

    assert len(pyramid) == 3
    assert pyramid[1.0].shape == (100, 100)
    assert pyramid[0.5].shape == (50, 50)
    assert pyramid[0.25].shape == (25, 25)
