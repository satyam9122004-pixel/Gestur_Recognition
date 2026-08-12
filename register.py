"""CLI utility to register a new person by capturing feature vectors from video."""
import argparse
import time

from vision.video_capture import VideoCapture
from vision.detector import PersonDetector
from features.feature_extractor import FeatureExtractor
from features.movement import MovementTracker
from database.database import Database


def register_person(name, video_source=0, num_samples=30, db_path="database/people.db"):
    detector = PersonDetector()
    extractor = FeatureExtractor()
    tracker = MovementTracker()
    db = Database(db_path)

    person = db.get_person_by_name(name)
    person_id = person["id"] if person else db.add_person(name)

    collected = 0
    with VideoCapture(video_source) as cap:
        # Warm up the background model on a handful of frames first.
        for _ in range(30):
            frame = cap.read()
            if frame is None:
                break
            detector.detect(frame)

        while collected < num_samples:
            frame = cap.read()
            if frame is None:
                break

            detections = detector.detect(frame)
            if not detections:
                continue

            # Assume the single largest detection is the person being registered.
            detection = max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
            tracker.update(person_id, detection.centroid, time.time())

            result = extractor.extract(detection, movement_tracker=tracker, track_id=person_id)
            db.add_feature_vector(person_id, result["vector"])
            collected += 1
            print(f"Collected sample {collected}/{num_samples}")

    db.close()
    print(f"Registered '{name}' with {collected} feature samples.")


def main():
    parser = argparse.ArgumentParser(description="Register a person's feature vectors into the database.")
    parser.add_argument("--name", required=True, help="Name of the person to register")
    parser.add_argument("--video", default=0, help="Video source (camera index or file path)")
    parser.add_argument("--samples", type=int, default=30, help="Number of feature samples to collect")
    parser.add_argument("--db", default="database/people.db", help="Path to the SQLite database file")
    args = parser.parse_args()

    video_source = int(args.video) if str(args.video).isdigit() else args.video
    register_person(args.name, video_source=video_source, num_samples=args.samples, db_path=args.db)


if __name__ == "__main__":
    main()
