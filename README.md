# Real-Time AR Guitar Instruction via Homographic Perspective Mapping
 
A web-based guitar chord trainer that uses your webcam and MediaPipe hand landmark detection to track your fretting hand in real time. Select a chord on the sidebar to see its diagram and overlay the chord name on the live video feed.

## Features

- Live webcam feed with hand landmark and fingertip overlays
- Chord diagram for G, C, D, E, A, Em, Am, F
- Chord name overlaid on the video frame
- Works in all browsers (Safari, Chrome, Firefox)

## Requirements

- Python 3.9+
- Webcam (built-in or external)

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/madihuang8/CS131-Final.git
   cd CS131-Final
   ```

2. Install dependencies:
   ```bash
   pip install flask opencv-python mediapipe
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. Open your browser and go to `http://127.0.0.1:5000`

## Running standalone (no browser)

To open a plain OpenCV window without the web interface:
```bash
python guitar.py
```
Press `q` or `ESC` to quit.

## Camera selection

By default the app uses camera index `1` (built-in FaceTime HD on Mac). If you need to change this, edit the top of `guitar.py`:

```python
CAMERA_INDEX = 1  # 0 = iPhone (Continuity Camera), 1 = built-in FaceTime
```

## Project structure

```
├── app.py              # Flask server
├── guitar.py           # Camera capture and MediaPipe hand detection
├── hand_landmarker.task  # MediaPipe hand landmark model
└── templates/
    └── index.html      # Frontend UI
```

## How it works

`guitar.py` runs hand detection in a background thread using MediaPipe's `HandLandmarker` in VIDEO mode. It draws skeleton connections and numbered fingertip dots onto each frame and stores the result in a shared `outputFrame`. The Flask server in `app.py` serves individual JPEG snapshots from that shared frame to the browser, which renders them on a canvas at ~30 fps.
