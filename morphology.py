"""Morphological post-processing for foreground masks."""
import cv2
import numpy as np


class MorphologyProcessor:
    def __init__(self, kernel_size=5, open_iterations=2, close_iterations=2, dilate_iterations=1):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.open_iterations = open_iterations
        self.close_iterations = close_iterations
        self.dilate_iterations = dilate_iterations

    def clean(self, mask):
        """Remove small noise (opening) then fill small gaps (closing)."""
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iterations)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iterations)
        if self.dilate_iterations > 0:
            cleaned = cv2.dilate(cleaned, self.kernel, iterations=self.dilate_iterations)
        return cleaned

    def fill_holes(self, mask):
        """Fill interior holes in blobs by redrawing filled external contours."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled
