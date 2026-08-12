"""Geometric shape descriptors computed from a contour."""
import cv2


def compute_shape_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / float(w) if w > 0 else 0.0
    extent = area / float(w * h) if w * h > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area) if hull_area > 0 else 0.0

    circularity = (4 * 3.14159265 * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "width": float(w),
        "height": float(h),
    }
