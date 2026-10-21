# Finger Counter

A real-time Python app that uses OpenCV and MediaPipe to count the raised
fingers on one or two hands.

## Run

```powershell
.\.venv\Scripts\python.exe .\main.py
```

Show one or two hands to the camera. The app displays each hand's count and a
stable total from 0 to 10.
Press `q` to quit.

## Project structure

- `main.py`: camera loop and interface
- `app/tracker.py`: MediaPipe hand detection and finger-state logic
- `models/hand_landmarker.task`: trained MediaPipe model
- `app/canvas.py`: earlier air-writing experiment; not used by the counter
