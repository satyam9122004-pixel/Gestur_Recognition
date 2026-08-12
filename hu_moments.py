"""Hu moment invariants for shape matching."""
import cv2
import numpy as np


def compute_hu_moments(contour_or_mask):
    """Compute the 7 log-scaled Hu moment invariants.

    Accepts either a contour (ndarray of points) or a binary mask image.
    """
    moments = cv2.moments(contour_or_mask)
    hu = cv2.HuMoments(moments).flatten()
    # Log-scale for numerical stability (Hu moments span many orders of magnitude)
    with np.errstate(divide="ignore"):
        log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return log_hu
