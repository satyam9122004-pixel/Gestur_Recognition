"""Combine shape features, Hu moments, and movement into one feature vector."""
import numpy as np

from features.shape_features import compute_shape_features
from features.hu_moments import compute_hu_moments


class FeatureExtractor:
    # Scale-dependent features (area, perimeter, width, height) are deliberately
    # excluded from the vector so matching is robust to a person's distance
    # from the camera.
    FEATURE_ORDER = [
        "aspect_ratio", "extent", "solidity", "circularity",
    ]

    def extract(self, detection, movement_tracker=None, track_id=None):
        """Build a feature vector for a single Detection.

        Returns a dict with 'vector' (numpy array) and 'raw' (all named features)
        so callers can store either the compact vector or the full breakdown.
        """
        shape_feats = compute_shape_features(detection.contour)
        hu = compute_hu_moments(detection.contour)

        speed = 0.0
        if movement_tracker is not None and track_id is not None:
            speed = movement_tracker.get_speed(track_id)

        vector_parts = [shape_feats[name] for name in self.FEATURE_ORDER]
        vector_parts.extend(hu.tolist())
        vector_parts.append(speed)

        vector = np.array(vector_parts, dtype=np.float64)

        raw = dict(shape_feats)
        raw["hu_moments"] = hu.tolist()
        raw["speed"] = speed

        return {"vector": vector, "raw": raw}
