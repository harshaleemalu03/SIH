import cv2
import numpy as np


def resize_image(image, scale):
    """
    Resize an image by a given scale factor.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    scale : float
        Scale factor.
        < 1.0 -> downsampling
        > 1.0 -> upsampling

    Returns
    -------
    numpy.ndarray
        Rescaled image.
    """

    if scale <= 0:
        raise ValueError("Scale must be greater than zero.")

    height, width = image.shape[:2]

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_CUBIC
    )


def simulate_resolution_change(image, scale):
    """
    Simulate lower spatial resolution by downsampling
    and then returning the lower-resolution image.

    This is useful for controlled scale experiments.
    """

    return resize_image(image, scale)


def scale_pyramid(
    image,
    scales=(1.0, 0.5, 0.25, 0.125)
):
    """
    Generate multiple scale versions of an image.

    Returns
    -------
    dict
        Mapping scale factor -> image.
    """

    return {
        scale: resize_image(image, scale)
        for scale in scales
    }
