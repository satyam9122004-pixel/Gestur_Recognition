# Person Identification System

A classical computer-vision pipeline that detects a person in video, describes
their silhouette with hand-crafted shape features, and matches that
description against a small database of previously registered people.

No deep learning is involved — detection relies on **background
subtraction**, **morphological filtering**, and **contour analysis**, and
identification relies on **Hu moment invariants**, **geometric shape
descriptors**, and **movement/speed**. This makes the system lightweight,
fully explainable, and easy to run without a GPU, at the cost of being far
less robust than a learned re-identification embedding.

## How it works

```
Video frame
   │
   ▼
Background subtraction (MOG2/KNN)  →  raw foreground mask
   │
   ▼
Morphological open/close + hole fill  →  clean blob mask
   │
   ▼
Contour extraction + filtering (area, aspect ratio)  →  candidate person blobs
   │
   ▼
Feature extraction (shape descriptors + Hu moments + speed)  →  feature vector
   │
   ▼
Matching (nearest-neighbor distance vs. database)  →  identity or "unknown"
```

### Vision (`vision/`)
- **`video_capture.py`** — thin wrapper around `cv2.VideoCapture` for webcams or video files.
- **`background.py`** — MOG2/KNN background subtractor that yields a binary foreground mask.
- **`morphology.py`** — opening/closing/dilation to remove noise and fill small gaps, plus a hole-filling step.
- **`contours.py`** — extracts external contours and filters them by area and by height/width aspect ratio (a proxy for "looks like a standing person").
- **`detector.py`** — orchestrates the above into a `PersonDetector.detect(frame) -> list[Detection]`.

### Features (`features/`)
- **`shape_features.py`** — aspect ratio, extent, solidity, circularity from the contour.
- **`hu_moments.py`** — the 7 log-scaled Hu moment invariants (translation/scale/rotation invariant shape descriptors).
- **`movement.py`** — tracks a blob's centroid over time and estimates speed, since gait/movement speed is a weak but useful cue.
- **`feature_extractor.py`** — combines the above into a single numeric feature vector per detection.

### Database (`database/`)
- **`database.py`** — SQLite wrapper (`people` and `features` tables) for storing registered identities and their sampled feature vectors.
- **`register.py`** — CLI script that captures video, collects N feature samples for a named person, and stores them.
- **`people.db`** — created automatically on first run; not checked in with data by default.

### Matching (`matching/`)
- **`distance.py`** — Euclidean, cosine, and weighted-Euclidean distance functions.
- **`matcher.py`** — compares a query feature vector against all stored vectors (nearest-neighbor per person) and returns the best match, or `"unknown"` if the closest match exceeds a distance threshold.

### Integration (`integration/`)
- **`pipeline.py`** — wires capture → detect → extract → match → annotate into a live loop with on-screen bounding boxes and labels.

### `main.py`
CLI entry point with two subcommands: `register` and `run`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Register a person

Stand in front of the camera and run:

```bash
python main.py register --name "Satyam" --video 0 --samples 30
```

This collects 30 feature samples of the largest detected blob and stores them
in `database/people.db` under the given name. Use `--video path/to/file.mp4`
to register from a recorded clip instead of a live camera.

### 2. Run live identification

```bash
python main.py run --video 0 --threshold 15.0
```

This opens a window showing bounding boxes and predicted names for every
detected person, drawing `"unknown"` for anyone whose closest match exceeds
`--threshold`. Press `q` to quit. Use `--no-display` to run headless.

## Tests

```bash
pytest tests/
```

Tests use synthetic masks/contours (no camera or real video required), so
they run in CI or any offline environment.

## Design notes & limitations

- **Not re-identification-grade.** Hu moments and simple shape ratios are
  sensitive to pose, occlusion, and clothing changes. This system is best
  suited for small, controlled scenarios (e.g., a handful of known people in
  a fixed-camera room) rather than open-world re-ID.
- **Single camera / single background model.** The background subtractor
  assumes a mostly static camera and background; it will need to re-adapt
  after camera motion or lighting changes.
- **Scale-invariance.** The feature vector intentionally excludes raw area,
  perimeter, width, and height so that matching does not depend on how far a
  person is standing from the camera; aspect ratio, extent, solidity,
  circularity, and Hu moments are all scale-invariant or normalized.
- **Extending the system.** Natural next steps: multi-frame temporal
  smoothing of features before matching, per-feature normalization/weighting
  in `matching/distance.py`, or swapping in a learned embedding as a drop-in
  replacement for `features/feature_extractor.py` while keeping the rest of
  the pipeline unchanged.

## Project structure

```
person-identification-system/
│
├── README.md
├── requirements.txt
├── main.py
│
├── vision/            # capture, background subtraction, morphology, contours, detector
├── features/           # shape features, Hu moments, movement, feature extractor
├── database/            # SQLite storage + registration CLI
├── matching/            # distance metrics + nearest-neighbor matcher
├── integration/          # end-to-end pipeline
├── tests/               # unit tests (synthetic data, no camera needed)
└── data/videos/          # place sample video files here for offline testing
```
