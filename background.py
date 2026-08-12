"""Background subtraction module."""
import cv2


class BackgroundSubtractor:
    def __init__(self, method="MOG2", history=500, var_threshold=16, detect_shadows=True):
        self.method = method
        if method == "MOG2":
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows
            )
        elif method == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history, detectShadows=detect_shadows
            )
        else:
            raise ValueError(f"Unknown background subtraction method: {method}")

    def apply(self, frame, learning_rate=-1):
        mask = self.subtractor.apply(frame, learningRate=learning_rate)
        # Remove shadow pixels (value 127 in MOG2/KNN shadow detection)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return mask

    def get_background_image(self):
        return self.subtractor.getBackgroundImage()
