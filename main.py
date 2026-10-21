from app.tracker import HandTracker
from app.canvas import Canvas
import cv2


CAMERA_INDEX = 0
WINDOW_NAME = "Air Writing App"
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
DRAW_COLOR = (0, 255, 0)
DRAW_THICKNESS = 6

camera = cv2.VideoCapture(CAMERA_INDEX)
tracker = HandTracker()
canvas = Canvas(CANVAS_WIDTH, CANVAS_HEIGHT)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,CANVAS_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,CANVAS_HEIGHT)


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
        frame = cv2.resize(
            frame,
            (CANVAS_WIDTH, CANVAS_HEIGHT),
        )
        hand_detected = tracker.process(frame)

        if hand_detected:
            tracker.draw_skeleton(frame)
            tip = tracker.position(
                HandTracker.INDEX_TIP,
                frame.shape,
            )
            fingers = tracker.fingers_up()

            if fingers == [True, False, False, False]:
                canvas.draw(
                    tip,
                    DRAW_COLOR,
                    DRAW_THICKNESS,
                )
            else:
                canvas.lift_pen()

            cv2.circle(
                frame,
                tip,
                12,
                (0, 255, 0),
                -1,
            )
            print(tracker.fingers_up())

            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else :
            canvas.lift_pen()


        output = canvas.merge(frame)
        cv2.imshow(WINDOW_NAME, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas.clear()
finally:
    camera.release()
    tracker.close()
    cv2.destroyAllWindows()
