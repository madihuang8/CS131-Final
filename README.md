# Real-Time AR Guitar Instruction
 
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
├── fretboard.py        # Fretboard mapping and augmentation
├── guitar.py           # Camera capture and MediaPipe hand detection
├── hand_landmarker.task  # MediaPipe hand landmark model
└── templates/
    └── index.html      # Frontend UI
```

## How it works

`guitar.py` runs camera capture and hand detection in a background thread using MediaPipe’s HandLandmarker in VIDEO mode. It draws hand skeleton connections and numbered fingertip dots onto each frame, then stores the annotated result in a shared outputFrame.

`fretboard.py` handles fretboard detection and AR augmentation. It uses OpenCV image processing, including Canny edge detection and Hough line transforms, to identify candidate fretboard, string, and fret lines. Once the fretboard is locked, it estimates the fret/string geometry and projects chord finger placements onto the live video feed.

The Flask server in `app.py` serves individual JPEG snapshots from the shared frame to the browser. The frontend in templates/index.html renders those snapshots on a canvas at approximately 30 fps and provides UI controls for locking/resetting the fretboard and selecting chords.
