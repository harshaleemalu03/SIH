import cv2


def detect_and_match(image1, image2, ratio=0.75):
    """
    Detect SIFT features and perform descriptor matching.

    Parameters
    ----------
    image1 : numpy.ndarray
        Source grayscale image.

    image2 : numpy.ndarray
        Reference grayscale image.

    ratio : float
        Lowe's ratio-test threshold.

    Returns
    -------
    keypoints1
    keypoints2
    good_matches
    """

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(
        image1, None
    )

    keypoints2, descriptors2 = sift.detectAndCompute(
        image2, None
    )

    if descriptors1 is None or descriptors2 is None:
        return keypoints1, keypoints2, []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )

    raw_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in raw_matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches
