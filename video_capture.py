"""Video capture utility wrapping OpenCV VideoCapture."""
import cv2


class VideoCapture:
    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
