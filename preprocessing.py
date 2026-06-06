"""Shared preprocessing for ASL recognition.

This is the SINGLE SOURCE OF TRUTH for how a grayscale hand crop becomes a
feature vector. Both app.py (live inference) and feature_extraction.ipynb
(training) must import from here so the two pipelines can never drift apart.

In the notebook, replace the inline square_pad / hog calls with:

    from preprocessing import extract_features
    features_hog = extract_features(gray_crop)
"""

import cv2
from skimage.feature import hog

# Resolution the model was trained at (HOG vector length depends on this).
IMAGE_SIZE = (128, 128)

# HOG configuration. Changing any of these requires re-extracting features
# AND retraining the model, since it changes the feature-vector length/meaning.
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=False,
)


def square_pad(crop):
    """Pad a grayscale crop to a square with black borders.

    Prevents aspect-ratio distortion when the crop is later resized to a
    square, which would otherwise warp two-finger letters (U/V/R/K).
    """
    h, w = crop.shape
    s = max(h, w)
    top = (s - h) // 2
    bottom = s - h - top
    left = (s - w) // 2
    right = s - w - left
    return cv2.copyMakeBorder(
        crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )


def extract_features(gray_crop):
    """grayscale crop -> square_pad -> resize -> HOG feature vector."""
    padded = square_pad(gray_crop)
    resized = cv2.resize(padded, IMAGE_SIZE)
    return hog(resized, **HOG_PARAMS)
