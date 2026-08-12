"""Tests for distance metrics and the person matcher."""
from matching.distance import euclidean_distance, cosine_distance
from matching.matcher import PersonMatcher


class FakeDatabase:
    """Minimal stand-in for the Database class used only in matcher tests."""

    def __init__(self, records):
        # records: list of (person_id, name, vector)
        self.records = records

    def get_feature_vectors(self, person_id=None):
        if person_id is None:
            return self.records
        return [r for r in self.records if r[0] == person_id]


def test_euclidean_distance_zero_for_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert euclidean_distance(v, v) == 0.0


def test_cosine_distance_zero_for_identical_direction():
    v = [1.0, 2.0, 3.0]
    assert cosine_distance(v, v) < 1e-9


def test_matcher_returns_closest_person():
    db = FakeDatabase([
        (1, "Alice", [0.0, 0.0, 0.0]),
        (2, "Bob", [10.0, 10.0, 10.0]),
    ])
    matcher = PersonMatcher(db, distance_fn=euclidean_distance, threshold=5.0)
    result = matcher.match([0.5, 0.5, 0.5])
    assert result.name == "Alice"


def test_matcher_returns_unknown_beyond_threshold():
    db = FakeDatabase([
        (1, "Alice", [0.0, 0.0, 0.0]),
    ])
    matcher = PersonMatcher(db, distance_fn=euclidean_distance, threshold=1.0)
    result = matcher.match([50.0, 50.0, 50.0])
    assert result.name == "unknown"
