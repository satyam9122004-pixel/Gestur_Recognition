"""Tests for shape features, Hu moments, and the feature extractor."""
import numpy as np
import cv2

from features.shape_features import compute_shape_features
from features.hu_moments import compute_hu_moments
from features.feature_extractor import FeatureExtractor
from vision.detector import Detection


def _rect_contour(w=60, h=140):
    mask = np.zeros((h + 20, w + 20), dtype=np.uint8)
    mask[10:10 + h, 10:10 + w] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def test_shape_features_aspect_ratio():
    contour = _rect_contour(w=60, h=140)
    feats = compute_shape_features(contour)
    assert abs(feats["aspect_ratio"] - (140 / 60)) < 0.2


def test_shape_features_extent_near_one_for_rectangle():
    contour = _rect_contour(w=60, h=140)
    feats = compute_shape_features(contour)
    assert feats["extent"] > 0.9


def test_hu_moments_returns_seven_values():
    contour = _rect_contour()
    hu = compute_hu_moments(contour)
    assert hu.shape == (7,)


def test_feature_extractor_vector_length():
    contour = _rect_contour()
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    detection = Detection(contour, (x, y, w, h), mask)

    extractor = FeatureExtractor()
    result = extractor.extract(detection)
    # 4 shape features + 7 Hu moments + 1 speed value
    assert result["vector"].shape == (12,)
