import cv2


def draw_matches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches,
    mask=None
):
    """
    Draw feature matches between two images.

    If a RANSAC mask is supplied, only inlier matches
    are displayed.
    """

    matches_mask = None

    if mask is not None:
        matches_mask = mask.ravel().tolist()

    return cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
