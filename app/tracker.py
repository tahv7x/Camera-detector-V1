import time
from pathlib import Path

import cv2
import mediapipe as mp


class HandTracker:
    """Detect one hand and report which fingers are raised."""

    THUMB_TIP = 4
    THUMB_IP = 3
    INDEX_TIP = 8
    FINGER_TIPS = (8, 12, 16, 20)
    FINGER_JOINTS = (
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20),
    )
    FINGER_EXTENSION_RATIOS = (1.10, 1.10, 1.08, 1.04)
    THUMB_SPREAD_RATIO = 0.70

    def __init__(self, model_path=None):
        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent
            model_path = project_root / "models" / "hand_landmarker.task"

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Hand model not found: {model_path}")

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self._landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )
        self._result = None
        self._last_timestamp_ms = -1

    def process(self, frame_bgr):
        """Process one OpenCV frame and return True when a hand is found."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        self._result = self._landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )
        return bool(self._result.hand_landmarks)

    @property
    def hand_label(self):
        """Return Right/Left for the detected hand, or None."""
        if not self._has_hand():
            return None
        raw_label = self._result.handedness[0][0].category_name
        return self._physical_hand_label(raw_label)

    @property
    def hand_labels(self):
        """Return the handedness labels for every detected hand."""
        if not self._has_hand():
            return []
        return [
            self._physical_hand_label(categories[0].category_name)
            for categories in self._result.handedness
        ]

    @property
    def hand_count(self):
        """Return how many hands are currently detected."""
        if not self._has_hand():
            return 0
        return len(self._result.hand_landmarks)

    def position(self, index, frame_shape, hand_index=0):
        """Return a landmark position in frame pixels."""
        if not self._has_hand():
            return None

        height, width = frame_shape[:2]
        landmark = self._result.hand_landmarks[hand_index][index]
        return int(landmark.x * width), int(landmark.y * height)

    def fingers_up(self, hand_index=0):
        """Return [thumb, index, middle, ring, pinky] as booleans."""
        if not self._has_hand():
            return [False] * 5

        landmarks = self._result.hand_landmarks[hand_index]

        palm_width = max(
            self._distance(landmarks[5], landmarks[17]),
            1e-6,
        )
        thumb_spread = (
            self._distance(landmarks[4], landmarks[5]) / palm_width
        )
        thumb_up = thumb_spread >= self.THUMB_SPREAD_RATIO

        states = [thumb_up]
        wrist = landmarks[0]
        for (_mcp, pip, tip), extension_ratio in zip(
            self.FINGER_JOINTS,
            self.FINGER_EXTENSION_RATIOS,
        ):
            tip_distance = self._distance(landmarks[tip], wrist)
            joint_distance = self._distance(landmarks[pip], wrist)
            states.append(
                tip_distance
                >= joint_distance * extension_ratio
            )

        return states

    def count_fingers(self, hand_index=0):
        """Return the number of raised fingers, from 0 to 5."""
        return sum(self.fingers_up(hand_index))

    def draw_skeleton(self, frame):
        """Draw the detected hand landmarks over an OpenCV frame."""
        if not self._has_hand():
            return

        point_style = mp.tasks.vision.drawing_utils.DrawingSpec(
            color=(0, 80, 255),
            thickness=2,
            circle_radius=3,
        )
        line_style = mp.tasks.vision.drawing_utils.DrawingSpec(
            color=(255, 255, 255),
            thickness=2,
        )

        for hand_landmarks in self._result.hand_landmarks:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=(
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                ),
                landmark_drawing_spec=point_style,
                connection_drawing_spec=line_style,
            )

    def _has_hand(self):
        return bool(self._result and self._result.hand_landmarks)

    @staticmethod
    def _physical_hand_label(raw_label):
        """Correct MediaPipe handedness after mirroring the camera frame."""
        if raw_label == "Left":
            return "Right"
        if raw_label == "Right":
            return "Left"
        return raw_label

    @staticmethod
    def _distance(first, second):
        return (
            (first.x - second.x) ** 2
            + (first.y - second.y) ** 2
        ) ** 0.5

    def close(self):
        self._landmarker.close()
