"""Tests for the vision detection pipeline using synthetic frames."""
import numpy as np

from vision.morphology import MorphologyProcessor
from vision.contours import find_contours, filter_contours


def _make_blob_mask(shape=(200, 200), rect=(70, 30, 60, 140)):
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, w, h = rect
    mask[y:y + h, x:x + w] = 255
    return mask


def test_morphology_clean_preserves_blob():
    processor = MorphologyProcessor(kernel_size=3, open_iterations=1, close_iterations=1, dilate_iterations=0)
    mask = _make_blob_mask()
    cleaned = processor.clean(mask)
    assert cleaned.sum() > 0


def test_find_contours_detects_blob():
    mask = _make_blob_mask()
    contours = find_contours(mask)
    assert len(contours) == 1


def test_filter_contours_by_aspect_ratio():
    mask = _make_blob_mask(rect=(70, 30, 60, 140))  # tall, person-like: h/w ~2.3
    contours = find_contours(mask)
    filtered = filter_contours(contours, min_area=100, min_aspect_ratio=1.0, max_aspect_ratio=5.0)
    assert len(filtered) == 1


def test_filter_contours_rejects_wide_blob():
    mask = _make_blob_mask(rect=(20, 80, 160, 40))  # wide, not person-like: h/w = 0.25
    contours = find_contours(mask)
    filtered = filter_contours(contours, min_area=100, min_aspect_ratio=1.0, max_aspect_ratio=5.0)
    assert len(filtered) == 0
