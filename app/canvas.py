import cv2
import numpy as np


class Canvas:
    def __init__(self, width, height):
        self.layer = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        self._current_point = None

    def draw(self, point, color, thickness):
        if self._current_point is not None:
            cv2.line(
                self.layer,
                self._current_point,
                point,
                color,
                thickness,
                cv2.LINE_AA,
            )

        self._current_point = point

    def lift_pen(self):
        self._current_point = None

    def clear(self):
        self.layer[:] = 0
        self.lift_pen()

    def merge(self, frame):
        gray = cv2.cvtColor(
            self.layer,
            cv2.COLOR_BGR2GRAY,
        )

        mask = gray > 0
        output = frame.copy()
        output[mask] = self.layer[mask]

        return output