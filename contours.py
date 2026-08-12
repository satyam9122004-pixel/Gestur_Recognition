"""Contour extraction and filtering utilities."""
import cv2


def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours(contours, min_area=1500, max_area=None, min_aspect_ratio=1.0, max_aspect_ratio=5.0):
    """Filter contours to keep those likely to be a standing/moving person.

    Person silhouettes are typically taller than they are wide, so we
    filter by area and by bounding-box aspect ratio (height / width).
    """
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w == 0:
            continue
        aspect_ratio = h / float(w)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        filtered.append(contour)
    return filtered


def largest_contour(contours):
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)
