"""Match a query feature vector against people stored in the database."""
from matching.distance import euclidean_distance


class MatchResult:
    def __init__(self, person_id, name, distance):
        self.person_id = person_id
        self.name = name
        self.distance = distance

    def __repr__(self):
        return f"MatchResult(name={self.name!r}, distance={self.distance:.4f})"


class PersonMatcher:
    def __init__(self, db, distance_fn=euclidean_distance, threshold=15.0):
        """
        db: a Database instance with stored feature vectors.
        distance_fn: function(v1, v2) -> float, lower means more similar.
        threshold: max distance to consider a match valid (else "unknown").
        """
        self.db = db
        self.distance_fn = distance_fn
        self.threshold = threshold

    def match(self, query_vector, person_id=None):
        """Compare query_vector against all stored vectors and return the best match.

        Since each person may have multiple stored samples, we take the
        minimum distance across all of a person's samples (nearest-neighbor).
        """
        stored = self.db.get_feature_vectors(person_id=person_id)
        if not stored:
            return None

        best_by_person = {}
        for pid, name, vector in stored:
            dist = self.distance_fn(query_vector, vector)
            if pid not in best_by_person or dist < best_by_person[pid].distance:
                best_by_person[pid] = MatchResult(pid, name, dist)

        best_match = min(best_by_person.values(), key=lambda m: m.distance)

        if best_match.distance > self.threshold:
            return MatchResult(None, "unknown", best_match.distance)

        return best_match
