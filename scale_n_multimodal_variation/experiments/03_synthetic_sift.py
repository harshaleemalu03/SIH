import cv2
import numpy as np

from src.matching.sift import detect_and_match
from src.registration.geometric import estimate_homography
from src.evaluation.metrics import (
    inlier_count,
    inlier_ratio,
    reprojection_rmse
)
from src.evaluation.visualization import draw_matches


def create_test_image():

    image = np.zeros(
        (600, 600),
        dtype=np.uint8
    )

    cv2.circle(
        image,
        (180, 180),
        70,
        255,
        3
    )

    cv2.circle(
        image,
        (420, 350),
        100,
        200,
        4
    )

    cv2.rectangle(
        image,
        (100, 400),
        (250, 520),
        180,
        3
    )

    cv2.line(
        image,
        (300, 80),
        (500, 200),
        220,
        4
    )

    cv2.putText(
        image,
        "LUNAR",
        (250, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        255,
        3
    )

    return image


def main():

    source = create_test_image()

    rotation_matrix = cv2.getRotationMatrix2D(
        (300, 300),
        12,
        1.0
    )

    reference = cv2.warpAffine(
        source,
        rotation_matrix,
        (600, 600)
    )

    kp1, kp2, matches = detect_and_match(
        source,
        reference
    )

    print("Keypoints in source:", len(kp1))
    print("Keypoints in reference:", len(kp2))
    print("Good matches:", len(matches))

    H, mask = estimate_homography(
        kp1,
        kp2,
        matches
    )

    if H is None:
        print("Homography estimation failed.")
        return

    print("Inliers:", inlier_count(mask))
    print("Inlier ratio:", inlier_ratio(mask))

    rmse = reprojection_rmse(
        kp1,
        kp2,
        matches,
        H,
        mask
    )

    print("Reprojection RMSE:", rmse)

    result = draw_matches(
        source,
        kp1,
        reference,
        kp2,
        matches,
        mask
    )

    cv2.imwrite(
        "results/matches/synthetic_sift.png",
        result
    )


if __name__ == "__main__":
    main()import cv2
import numpy as np

from src.matching.sift import detect_and_match
from src.registration.geometric import estimate_homography
from src.evaluation.metrics import (
    inlier_count,
    inlier_ratio,
    reprojection_rmse
)
from src.evaluation.visualization import draw_matches


def create_test_image():

    image = np.zeros(
        (600, 600),
        dtype=np.uint8
    )

    cv2.circle(
        image,
        (180, 180),
        70,
        255,
        3
    )

    cv2.circle(
        image,
        (420, 350),
        100,
        200,
        4
    )

    cv2.rectangle(
        image,
        (100, 400),
        (250, 520),
        180,
        3
    )

    cv2.line(
        image,
        (300, 80),
        (500, 200),
        220,
        4
    )

    cv2.putText(
        image,
        "LUNAR",
        (250, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        255,
        3
    )

    return image


def main():

    source = create_test_image()

    rotation_matrix = cv2.getRotationMatrix2D(
        (300, 300),
        12,
        1.0
    )

    reference = cv2.warpAffine(
        source,
        rotation_matrix,
        (600, 600)
    )

    kp1, kp2, matches = detect_and_match(
        source,
        reference
    )

    print("Keypoints in source:", len(kp1))
    print("Keypoints in reference:", len(kp2))
    print("Good matches:", len(matches))

    H, mask = estimate_homography(
        kp1,
        kp2,
        matches
    )

    if H is None:
        print("Homography estimation failed.")
        return

    print("Inliers:", inlier_count(mask))
    print("Inlier ratio:", inlier_ratio(mask))

    rmse = reprojection_rmse(
        kp1,
        kp2,
        matches,
        H,
        mask
    )

    print("Reprojection RMSE:", rmse)

    result = draw_matches(
        source,
        kp1,
        reference,
        kp2,
        matches,
        mask
    )

    cv2.imwrite(
        "results/matches/synthetic_sift.png",
        result
    )


if __name__ == "__main__":
    main()
