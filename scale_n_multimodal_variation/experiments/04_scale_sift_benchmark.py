import cv2
import numpy as np

from src.preprocessing.scale import scale_pyramid
from src.matching.sift import detect_and_match
from src.registration.geometric import estimate_homography
from src.evaluation.metrics import (
    inlier_count,
    inlier_ratio,
    reprojection_rmse
)
from src.evaluation.spatial import match_distribution


def create_test_image():

    image = np.zeros(
        (800, 800),
        dtype=np.uint8
    )

    # Circular crater-like structures
    for center, radius in [
        ((200, 200), 80),
        ((600, 220), 100),
        ((350, 550), 120),
        ((650, 600), 70)
    ]:
        cv2.circle(
            image,
            center,
            radius,
            220,
            3
        )

        cv2.circle(
            image,
            center,
            radius // 2,
            150,
            2
        )

    # Surface-like linear structures
    cv2.line(
        image,
        (50, 700),
        (750, 100),
        180,
        3
    )

    cv2.line(
        image,
        (100, 100),
        (700, 700),
        130,
        2
    )

    # Additional texture
    rng = np.random.default_rng(42)

    noise = rng.normal(
        0,
        15,
        image.shape
    ).astype(np.float32)

    image = np.clip(
        image.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)

    return image


def main():

    source = create_test_image()

    # Create a transformed reference image.
    matrix = cv2.getRotationMatrix2D(
        (400, 400),
        8,
        1.0
    )

    reference = cv2.warpAffine(
        source,
        matrix,
        (800, 800)
    )

    scales = (
        1.0,
        0.5,
        0.25,
        0.125
    )

    pyramid = scale_pyramid(
        source,
        scales
    )

    print(
        "scale, keypoints, matches, inliers, "
        "inlier_ratio, rmse, coverage, uniformity"
    )

    for scale, scaled_source in pyramid.items():

        # Resize reference to the same scale so that
        # this experiment isolates scale effects.
        scaled_reference = cv2.resize(
            reference,
            (
                scaled_source.shape[1],
                scaled_source.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )

        kp1, kp2, matches = detect_and_match(
            scaled_source,
            scaled_reference
        )

        H, mask = estimate_homography(
            kp1,
            kp2,
            matches
        )

        inliers = inlier_count(mask)
        ratio = inlier_ratio(mask)

        rmse = reprojection_rmse(
            kp1,
            kp2,
            matches,
            H,
            mask
        )

        coverage, counts, uniformity = (
            match_distribution(
                kp1,
                matches,
                scaled_source.shape
            )
        )

        print(
            f"{scale}, "
            f"{len(kp1)}, "
            f"{len(matches)}, "
            f"{inliers}, "
            f"{ratio:.4f}, "
            f"{rmse}, "
            f"{coverage:.4f}, "
            f"{uniformity:.4f}"
        )


if __name__ == "__main__":
    main()
