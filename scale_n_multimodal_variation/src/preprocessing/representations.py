import cv2
import numpy as np


def normalize(image):
    """
    Robustly normalize an image to the range [0, 1]
    using the 2nd and 98th percentiles.
    """
    image = image.astype(np.float32)

    low = np.nanpercentile(image, 2)
    high = np.nanpercentile(image, 98)

    image = np.clip(image, low, high)

    if high <= low:
        return np.zeros_like(image)

    return (image - low) / (high - low)


def gradient_magnitude(image):
    """
    Compute gradient magnitude using Sobel operators.
    Useful for testing structural representations
    that may be more stable across imaging modalities.
    """
    image = image.astype(np.float32)

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    return np.sqrt(gx * gx + gy * gy)
