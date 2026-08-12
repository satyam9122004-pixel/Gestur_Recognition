"""Person blob detector: combines background subtraction, morphology and contour filtering."""
import cv2

from vision.background import BackgroundSubtractor
from vision.morphology import MorphologyProcessor
from vision.contours import find_contours, filter_contours


class Detection:
    """A single detected foreground blob within a frame."""

    def __init__(self, contour, bbox, mask):
        self.contour = contour
        self.bbox = bbox  # (x, y, w, h)
        self.mask = mask  # cropped binary mask of the blob

    @property
    def centroid(self):
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


class PersonDetector:
    def __init__(self, min_area=1500, max_area=None, min_aspect_ratio=1.0, max_aspect_ratio=5.0,
                 bg_method="MOG2", kernel_size=5):
        self.bg_subtractor = BackgroundSubtractor(method=bg_method)
        self.morphology = MorphologyProcessor(kernel_size=kernel_size)
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def detect(self, frame):
        """Run the full detection pipeline on a single BGR frame.

        Returns a list of Detection objects.
        """
        fg_mask = self.bg_subtractor.apply(frame)
        clean_mask = self.morphology.clean(fg_mask)
        clean_mask = self.morphology.fill_holes(clean_mask)

        contours = find_contours(clean_mask)
        contours = filter_contours(
            contours,
            min_area=self.min_area,
            max_area=self.max_area,
            min_aspect_ratio=self.min_aspect_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
        )

        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            blob_mask = clean_mask[y:y + h, x:x + w]
            detections.append(Detection(contour, (x, y, w, h), blob_mask))
        return detections
