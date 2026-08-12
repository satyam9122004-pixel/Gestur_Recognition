"""Full end-to-end pipeline: capture -> detect -> extract features -> match -> annotate."""
import time
import cv2

from vision.video_capture import VideoCapture
from vision.detector import PersonDetector
from features.feature_extractor import FeatureExtractor
from features.movement import MovementTracker
from database.database import Database
from matching.matcher import PersonMatcher


class IdentificationPipeline:
    def __init__(self, db_path="database/people.db", video_source=0, threshold=15.0, display=True):
        self.detector = PersonDetector()
        self.extractor = FeatureExtractor()
        self.tracker = MovementTracker()
        self.db = Database(db_path)
        self.matcher = PersonMatcher(self.db, threshold=threshold)
        self.video_source = video_source
        self.display = display

    def _warm_up(self, cap, num_frames=30):
        for _ in range(num_frames):
            frame = cap.read()
            if frame is None:
                break
            self.detector.detect(frame)

    def run(self, max_frames=None):
        frame_count = 0
        with VideoCapture(self.video_source) as cap:
            self._warm_up(cap)

            while True:
                if max_frames is not None and frame_count >= max_frames:
                    break

                frame = cap.read()
                if frame is None:
                    break

                detections = self.detector.detect(frame)

                for i, detection in enumerate(detections):
                    track_id = i  # simple index-based tracking within this frame
                    self.tracker.update(track_id, detection.centroid, time.time())

                    result = self.extractor.extract(detection, movement_tracker=self.tracker, track_id=track_id)
                    match = self.matcher.match(result["vector"])

                    label = match.name if match else "unknown"
                    x, y, w, h = detection.bbox
                    if self.display:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)

                if self.display:
                    cv2.imshow("Person Identification", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1

        if self.display:
            cv2.destroyAllWindows()
        self.db.close()
