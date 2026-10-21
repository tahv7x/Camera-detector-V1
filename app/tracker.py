import mediapipe as mp
import time 
import cv2


class HandTracker:
    INDEX_TIP = 8
    def __init__(self, model_path="models/hand_landmarker.task"):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self._landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )
        self._result = None
    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = time.monotonic_ns() // 1_000_000

        self._result = self._landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )
        return bool(self._result.hand_landmarks)
    def position(self,index,frame_shape):
        height, width = frame_shape[:2]
        landmark = self._result.hand_landmarks[0][index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y
    def fingers_up(self):
        landmarks = self._result.hand_landmarks[0]
        stats = []

        for tip in [8,12,16,20]:
            joint = tip - 2
            is_up = landmarks[tip].y < landmarks[joint].y
            stats.append(is_up)
        return stats
    def draw_skeleton(self, frame):
        if self._result is None:
            return

        for hand_landmarks in self._result.hand_landmarks:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=(
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                ),
            )
    def close(self):
        self._landmarker.close()