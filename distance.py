"""Distance metrics for comparing feature vectors."""
import numpy as np


def euclidean_distance(v1, v2):
    v1, v2 = np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)
    return float(np.linalg.norm(v1 - v2))


def cosine_distance(v1, v2):
    v1, v2 = np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 1.0
    cosine_similarity = np.dot(v1, v2) / norm_product
    return float(1.0 - cosine_similarity)


def weighted_euclidean_distance(v1, v2, weights=None):
    v1, v2 = np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)
    diff = v1 - v2
    if weights is None:
        weights = np.ones_like(diff)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(np.sum(weights * diff ** 2)))
