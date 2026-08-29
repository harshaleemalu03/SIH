import cv2
import numpy as np

from src.matching.sift import detect_and_match


def test_sift_finds_matches():

    image = np.zeros(
        (300, 300),
        dtype=np.uint8
    )

    cv2.circle(
        image,
        (150, 150),
        60,
        255,
        -1
    )

    transformed = cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D(
            (150, 150),
            10,
            1.0
        ),
        (300, 300)
    )

    kp1, kp2, matches = detect_and_match(
        image,
        transformed
    )

    assert len(kp1) > 0
    assert len(kp2) > 0
    assert len(matches) > 0
