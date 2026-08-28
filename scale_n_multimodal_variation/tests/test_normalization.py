import numpy as np

from src.preprocessing.representations import normalize


def test_normalize_range():

    image = np.array(
        [
            [0, 10],
            [20, 30]
        ],
        dtype=np.float32
    )

    result = normalize(image)

    assert result.min() >= 0
    assert result.max() <= 1
