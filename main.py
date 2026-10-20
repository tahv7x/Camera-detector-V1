from app.tracker import HandTracker

import cv2


CAMERA_INDEX = 0
WINDOW_NAME = "Air Writing App"


camera = cv2.VideoCapture(CAMERA_INDEX)
tracker = HandTracker()

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")

print("Camera started. Press q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a frame from the camera.")
            break
        frame = cv2.flip(frame,1)
        hand_detected = tracker.process(frame)

        if hand_detected:
            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
finally:
    camera.release()
    tracker.close()
    cv2.destroyAllWindows()
