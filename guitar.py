import cv2
import mediapipe as mp
import time
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import fretboard

MODEL_PATH   = 'hand_landmarker.task'
CAMERA_INDEX = 1  # 0 = iPhone (Continuity Camera), 1 = built-in FaceTime

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

outputFrame = None
lock = threading.Lock()


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

            frame, detected_lines, edges = fretboard.detect_fretboard(frame)

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)

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
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

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
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)

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
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cam.release()
    cv2.destroyAllWindows()
