import cv2
import numpy as np


def detect_fretboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # reduce background noise
    edges = cv2.Canny(blurred, 50, 150) # we can tune this

    lines = cv2.HoughLinesP(edges,1, np.pi / 180,
        threshold=80, minLineLength=100, maxLineGap=15)

    output_frame = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output_frame, lines, edges


# local testing
if __name__ == '__main__':
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            drawn_frame, detected_lines, edge_mask = detect_fretboard(frame)

            cv2.imshow("Fretboard Lines", drawn_frame)
            cv2.imshow("Canny Edges", edge_mask)

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()