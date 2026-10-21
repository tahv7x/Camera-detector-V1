from collections import Counter, deque

import cv2

from app.tracker import HandTracker


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WINDOW_NAME = "Finger Counter"
SMOOTHING_WINDOW = 5


def stable_count(history, new_count):
    """Return the most common count from the latest frames."""
    history.append(new_count)
    return Counter(history).most_common(1)[0][0]


def draw_text_with_shadow(frame, text, position, scale, color, thickness):
    """Draw readable text without using an opaque background panel."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_interface(frame, total_count, hand_counts):
    """Draw a compact counter in the top-left corner."""
    number = "-" if total_count is None else str(total_count)
    draw_text_with_shadow(
        frame,
        number,
        (14, 58),
        1.8,
        (0, 255, 0),
        4,
    )

    if hand_counts:
        left_count = hand_counts.get("Left", "-")
        right_count = hand_counts.get("Right", "-")
        details = f"L:{left_count}  R:{right_count}"
    else:
        details = "No hands"

    draw_text_with_shadow(
        frame,
        details,
        (14, 84),
        0.55,
        (0, 220, 255),
        1,
    )


def main():
    tracker = HandTracker()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    count_histories = {
        "Left": deque(maxlen=SMOOTHING_WINDOW),
        "Right": deque(maxlen=SMOOTHING_WINDOW),
    }

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        if not camera.isOpened():
            raise RuntimeError("Could not open the camera.")

        print("Finger Counter started. Press q to quit.")

        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            total_count = None
            hand_counts = {}

            if tracker.process(frame):
                tracker.draw_skeleton(frame)

                for hand_index, hand_label in enumerate(tracker.hand_labels):
                    raw_count = tracker.count_fingers(hand_index)
                    hand_counts[hand_label] = stable_count(
                        count_histories[hand_label],
                        raw_count,
                    )

                for label, history in count_histories.items():
                    if label not in hand_counts:
                        history.clear()

                total_count = sum(hand_counts.values())
            else:
                for history in count_histories.values():
                    history.clear()

            draw_interface(frame, total_count, hand_counts)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
