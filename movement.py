"""Simple centroid-based movement/velocity tracking across frames."""
from collections import deque


class MovementTracker:
    def __init__(self, history_length=10):
        self.history_length = history_length
        self.tracks = {}  # track_id -> deque of (centroid, timestamp)

    def update(self, track_id, centroid, timestamp):
        if track_id not in self.tracks:
            self.tracks[track_id] = deque(maxlen=self.history_length)
        self.tracks[track_id].append((centroid, timestamp))

    def get_velocity(self, track_id):
        """Average velocity (dx, dy per second) over the stored history."""
        history = self.tracks.get(track_id)
        if not history or len(history) < 2:
            return (0.0, 0.0)

        (x0, y0), t0 = history[0]
        (x1, y1), t1 = history[-1]
        dt = t1 - t0
        if dt <= 0:
            return (0.0, 0.0)
        return ((x1 - x0) / dt, (y1 - y0) / dt)

    def get_speed(self, track_id):
        vx, vy = self.get_velocity(track_id)
        return (vx ** 2 + vy ** 2) ** 0.5

    def clear(self, track_id):
        self.tracks.pop(track_id, None)
