import cv2
import mediapipe as mp
import time
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import fretboard

MODEL_PATH   = 'hand_landmarker.task'
CAMERA_INDEX = 1  # 0 for iPhone, 1 for computer

# for the hand tracking
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGER_NAMES = {4: "1", 8: "2", 12: "3", 16: "4", 20: "5"}

CHORDS = {  # our chord dictionary
    'G':  {'frets': [3, 2, 0, 0, 0, 3], 'fingers': [2, 1, 0, 0, 0, 4]},
    'C':  {'frets': [-1, 3, 2, 0, 1, 0], 'fingers': [0, 3, 2, 0, 1, 0]},
    'D':  {'frets': [-1, -1, 0, 2, 3, 2], 'fingers': [0, 0, 0, 1, 3, 2]},
    'E':  {'frets': [0, 2, 2, 1, 0, 0], 'fingers': [0, 2, 3, 1, 0, 0]},
    'A':  {'frets': [-1, 0, 2, 2, 2, 0], 'fingers': [0, 0, 1, 2, 3, 0]},
    'Em': {'frets': [0, 2, 2, 0, 0, 0], 'fingers': [0, 2, 3, 0, 0, 0]},
    'Am': {'frets': [-1, 0, 2, 2, 1, 0], 'fingers': [0, 0, 2, 3, 1, 0]},
    'F':  {'frets': [1, 1, 2, 3, 3, 1], 'fingers': [1, 1, 2, 3, 4, 1]},
}


# 0.5 = center of fret box; higher values shift closer to the body-side fret wire.
FINGER_DOT_FRET_FRACTION = 0.72  # true to real guitar playing

REVERSE_STRING_ORDER = False  # based on our experience, this doesn't need to be TRUE

outputFrame = None
lock = threading.Lock()

def draw_chord_targets(frame, chord_name, fretboard_info):
    if not chord_name or chord_name not in CHORDS:
        return

    if not isinstance(fretboard_info, dict) or not fretboard_info.get('locked'):
        return

    fret_points = fretboard_info.get('tracked_frets')
    string_points = fretboard_info.get('tracked_strings')

    if fret_points is None or string_points is None:
        return

    fret_points = np.array(fret_points, dtype=np.float32).reshape(-1, 2)
    string_points = np.array(string_points, dtype=np.float32).reshape(-1, 2)

    num_fret_lines = len(fret_points) // 2
    num_strings = len(string_points) // 2

    if num_fret_lines < 2 or num_strings != 6:
        return

    chord = CHORDS[chord_name]
    frets = chord['frets']
    fingers = chord['fingers']

    for chord_string_index, fret_number in enumerate(frets):
        if fret_number <= 0:
            continue

        if fret_number >= num_fret_lines:
            continue

        tracked_string_index = 5 - chord_string_index if REVERSE_STRING_ORDER else chord_string_index

        string_start = string_points[2 * tracked_string_index]
        string_end = string_points[2 * tracked_string_index + 1]

        # shift the dot toward the body-side fret wire instead of centering it.
        t = ((fret_number - 1) + FINGER_DOT_FRET_FRACTION) / (num_fret_lines - 1)
        t = max(0.0, min(1.0, t))

        point = (1.0 - t) * string_start + t * string_end
        x, y = np.round(point).astype(int)

        cv2.circle(frame, (x, y), 16, (0, 200, 0), cv2.FILLED)
        cv2.circle(frame, (x, y), 16, (255, 255, 255), 2)

        finger_label = fingers[chord_string_index]
        if finger_label:
            cv2.putText(
                frame,
                str(finger_label),
                (x - 6, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

def _run_detection(state, state_lock):
    global outputFrame
    print("[guitar] detection thread started", flush=True)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print("[guitar] ERROR: cannot open camera", flush=True)
        return

    frame_width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[guitar] camera opened {frame_width}x{frame_height}", flush=True)

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            # run on the clean camera frame, not the frame with fretboard drawings
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)

            # draw fretboard annotations on a separate display frame
            try:
                display_frame, fretboard_info, edges = fretboard.detect_fretboard_with_labels(frame)
            except Exception as e:
                print("[guitar] fretboard labeling error:", repr(e), flush=True)
                display_frame, _, edges = fretboard.detect_fretboard(frame)
                fretboard_info = None

            frame = display_frame

            for hand_landmarks in result.hand_landmarks:
                for start, end in HAND_CONNECTIONS:
                    x1 = int(hand_landmarks[start].x * frame_width)
                    y1 = int(hand_landmarks[start].y * frame_height)
                    x2 = int(hand_landmarks[end].x * frame_width)
                    y2 = int(hand_landmarks[end].y * frame_height)
                    cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

                for tip_id in FINGERTIP_IDS:
                    lm = hand_landmarks[tip_id]
                    cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                    cv2.circle(frame, (cx, cy), 14, (99, 102, 241), cv2.FILLED)
                    cv2.putText(frame, FINGER_NAMES[tip_id], (cx - 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            with state_lock:
                chord_name = state.get('chord', '')
            draw_chord_targets(frame, chord_name, fretboard_info)

            if chord_name:
                cv2.putText(frame, chord_name, (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (99, 102, 241), 2)

            with lock:
                outputFrame = frame.copy()

    cam.release()


def get_snapshot():
    with lock:
        if outputFrame is None:
            return None
        flag, encoded = cv2.imencode('.jpg', outputFrame)
        return bytearray(encoded) if flag else None


def start(state, state_lock):
    t = threading.Thread(target=_run_detection, args=(state, state_lock))
    t.daemon = True
    t.start()


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                frame = None
            else:
                flag, encoded = cv2.imencode('.jpg', outputFrame)
                frame = bytearray(encoded) if flag else None

        if frame is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


if __name__ == '__main__':
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,)

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        raise RuntimeError("Cannot open camera — check index or permissions")

    frame_width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)

            try:
                display_frame, fretboard_info, edges = fretboard.detect_fretboard_with_labels(frame)
            except Exception as e:
                print("[guitar] fretboard labeling error:", repr(e), flush=True)
                display_frame, fretboard_info, edges = fretboard.detect_fretboard(frame)

            frame = display_frame

            for hand_landmarks in result.hand_landmarks:
                for start, end in HAND_CONNECTIONS:
                    x1 = int(hand_landmarks[start].x * frame_width)
                    y1 = int(hand_landmarks[start].y * frame_height)
                    x2 = int(hand_landmarks[end].x * frame_width)
                    y2 = int(hand_landmarks[end].y * frame_height)
                    cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

                for tip_id in FINGERTIP_IDS:
                    lm = hand_landmarks[tip_id]
                    cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                    cv2.circle(frame, (cx, cy), 14, (99, 102, 241), cv2.FILLED)
                    cv2.putText(frame, FINGER_NAMES[tip_id], (cx - 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Guitar Chord Trainer', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cam.release()
    cv2.destroyAllWindows()
