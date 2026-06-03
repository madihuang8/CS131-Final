import cv2
import numpy as np


def line_angle_deg(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def line_length(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)


def normalize_angle(angle):
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


class FretboardTracker:
    def __init__(self):
        self.locked = False
        self.prev_gray = None
        self.prev_features = None
        self.line_points = None
        self.string_points = None
        self.feature_mask = None

    def reset(self):
        self.locked = False
        self.prev_gray = None
        self.prev_features = None
        self.line_points = None
        self.string_points = None
        self.feature_mask = None

    def lock_from_info(self, gray, info):
        if gray is None or info is None:
            return False

        required = ["fret_positions", "axis", "perp", "fretboard_low", "fretboard_high", "string_positions", "string_low", "string_high"]
        if not all(key in info for key in required):
            return False

        points = []
        for pos in info["fret_positions"]:
            p_top = _point_from_projection(info["axis"], info["perp"], pos, info["fretboard_low"])
            p_bot = _point_from_projection(info["axis"], info["perp"], pos, info["fretboard_high"])
            points.append(p_top)
            points.append(p_bot)

        if len(points) < 4:
            return False

        line_points = np.array(points, dtype=np.float32).reshape(-1, 2)

        string_points = []
        for across in info["string_positions"]:
            p_left = _point_from_projection(info["axis"], info["perp"], info["string_low"], across)
            p_right = _point_from_projection(info["axis"], info["perp"], info["string_high"], across)
            string_points.append(p_left)
            string_points.append(p_right)

        if len(string_points) != 12:
            return False

        string_points = np.array(string_points, dtype=np.float32).reshape(-1, 2)

        mask = self._make_tracking_mask(gray.shape, line_points)

        features = cv2.goodFeaturesToTrack(gray, maxCorners=150, qualityLevel=0.01, minDistance=7, blockSize=7, mask=mask)

        if features is None or len(features) < 8:
            return False

        self.locked = True
        self.prev_gray = gray.copy()
        self.prev_features = features.astype(np.float32)
        self.line_points = line_points
        self.string_points = string_points
        self.feature_mask = mask
        return True

    def update(self, gray):
        if not self.locked or self.prev_gray is None or self.prev_features is None or self.line_points is None or self.string_points is None:
            return None

        next_features, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_features,
            None, winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        if next_features is None or status is None:
            self.reset()
            return None

        status = status.reshape(-1).astype(bool)
        old_good = self.prev_features.reshape(-1, 2)[status]
        new_good = next_features.reshape(-1, 2)[status]

        if len(old_good) < 8:
            self.reset()
            return None

        transform, inliers = cv2.estimateAffinePartial2D(old_good, new_good,
            method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000, confidence=0.99)

        if transform is None or inliers is None or int(inliers.sum()) < 6:
            self.reset()
            return None

        ones = np.ones((self.line_points.shape[0], 1), dtype=np.float32)
        homogeneous_line_points = np.hstack([self.line_points, ones])
        self.line_points = (transform @ homogeneous_line_points.T).T.astype(np.float32)

        ones = np.ones((self.string_points.shape[0], 1), dtype=np.float32)
        homogeneous_string_points = np.hstack([self.string_points, ones])
        self.string_points = (transform @ homogeneous_string_points.T).T.astype(np.float32)

        self.prev_gray = gray.copy()
        self.prev_features = new_good.reshape(-1, 1, 2).astype(np.float32)

        if len(self.prev_features) < 40:
            all_points = np.vstack([self.line_points, self.string_points])
            mask = self._make_tracking_mask(gray.shape, all_points)
            features = cv2.goodFeaturesToTrack(gray,
                maxCorners=150, qualityLevel=0.01, minDistance=7, blockSize=7, mask=mask)
            if features is not None and len(features) >= 8:
                self.prev_features = features.astype(np.float32)
                self.feature_mask = mask

        return self.line_points, self.string_points

    def _make_tracking_mask(self, shape, line_points):
        mask = np.zeros(shape, dtype=np.uint8)

        x_min = max(0, int(np.min(line_points[:, 0])) - 40)
        x_max = min(shape[1] - 1, int(np.max(line_points[:, 0])) + 40)
        y_min = max(0, int(np.min(line_points[:, 1])) - 40)
        y_max = min(shape[0] - 1, int(np.max(line_points[:, 1])) + 40)

        mask[y_min:y_max, x_min:x_max] = 255
        return mask


def _point_from_projection(axis, perp, along, across):
    return axis * along + perp * across


def _cluster_positions(values, cluster_gap):
    if len(values) == 0:
        return np.array([])

    values = np.array(sorted(values), dtype=np.float32)
    clustered = []
    cluster = [values[0]]

    for value in values[1:]:
        if abs(value - np.mean(cluster)) < cluster_gap:
            cluster.append(value)
        else:
            clustered.append(np.mean(cluster))
            cluster = [value]

    clustered.append(np.mean(cluster))
    return np.array(clustered, dtype=np.float32)


def estimate_six_string_positions(perp_values):
    if len(perp_values) == 0:
        return None

    clustered = _cluster_positions(perp_values, cluster_gap=18)
    if len(clustered) >= 6:
        # keep the six strongest central string-like bands.
        # under good framing, percentile trimming removes fretboard edges/background lines.
        if len(clustered) > 6:
            lo = np.percentile(clustered, 5)
            hi = np.percentile(clustered, 95)
            candidates = clustered[(clustered >= lo) & (clustered <= hi)]
            if len(candidates) >= 6:
                clustered = candidates

        if len(clustered) > 6:
            center = np.median(clustered)
            clustered = np.array(sorted(clustered, key=lambda x: abs(x - center))[:6])

        return np.sort(clustered[:6])

    if len(clustered) >= 2:
        # interpolation since it's difficult to detect all strings, especially the smaller ones
        # infer all six as equally spaced between detected extremes.
        return np.linspace(np.min(clustered), np.max(clustered), 6).astype(np.float32)

    return None


_tracker = FretboardTracker()
_last_gray = None
_last_info = None


def lock_current_fretboard():
    global _last_gray, _last_info
    if _last_gray is None or _last_info is None:
        return False
    return _tracker.lock_from_info(_last_gray, _last_info)


def reset_locked_fretboard():
    _tracker.reset()


def is_fretboard_locked():
    return _tracker.locked


def detect_fretboard_with_labels(frame, max_frets=12):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if _tracker.locked:
        output = frame.copy()
        tracked = _tracker.update(gray)
        if tracked is not None:
            tracked_frets, tracked_strings = tracked
            _draw_locked_fretboard(output, tracked_frets)
            _draw_locked_strings(output, tracked_strings)
            return output, {"locked": True, "tracked_frets": tracked_frets, "tracked_strings": tracked_strings}, gray

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
        threshold = 80, minLineLength = 80, maxLineGap = 20)

    output = frame.copy()
    if lines is None:
        return output, None, edges

    raw_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = normalize_angle(line_angle_deg(x1, y1, x2, y2))
        length = line_length(x1, y1, x2, y2)
        raw_lines.append({
            "p1": np.array([x1, y1], dtype=np.float32),
            "p2": np.array([x2, y2], dtype=np.float32),
            "angle": angle,
            "length": length,
        })

    # look at all detected lines that are >180 pixels in length
    long_lines = [l for l in raw_lines if l["length"] > 180]
    if len(long_lines) == 0:
        return output, lines, edges

    string_angle = np.median([l["angle"] for l in long_lines])
    theta = np.radians(string_angle)
    axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    perp = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)

    string_candidates = []
    fret_candidates = []
    string_position_values = []

    for l in raw_lines:
        angle_diff_string = abs(normalize_angle(l["angle"] - string_angle))
        angle_diff_fret = abs(abs(angle_diff_string) - 90)

        if angle_diff_string < 12 and l["length"] > 120:
            string_candidates.append(l)
            midpoint = (l["p1"] + l["p2"]) / 2
            string_position_values.append(float(midpoint @ perp))

        if angle_diff_fret < 18 and l["length"] > 40:
            fret_candidates.append(l)

    if len(string_candidates) < 2 or len(fret_candidates) < 3:
        return output, lines, edges

    string_positions = estimate_six_string_positions(string_position_values)
    if string_positions is None:
        return output, lines, edges

    string_midpoints = np.array([(l["p1"] + l["p2"]) / 2 for l in string_candidates])
    perp_values = string_midpoints @ perp

    low = np.percentile(perp_values, 10)
    high = np.percentile(perp_values, 90)

    pad = 25
    fretboard_low = low - pad
    fretboard_high = high + pad

    fret_positions = []
    for l in fret_candidates:
        midpoint = (l["p1"] + l["p2"]) / 2
        along = float(midpoint @ axis)
        across = float(midpoint @ perp)

        if fretboard_low <= across <= fretboard_high:
            fret_positions.append(along)

    if len(fret_positions) < 3:
        return output, lines, edges

    fret_positions = np.array(sorted(fret_positions))

    clustered = []
    cluster = [fret_positions[0]]
    cluster_gap = 25

    for pos in fret_positions[1:]:
        if abs(pos - np.mean(cluster)) < cluster_gap:
            cluster.append(pos)
        else:
            clustered.append(np.mean(cluster))
            cluster = [pos]

    clustered.append(np.mean(cluster))
    fret_positions = np.array(clustered)
    fret_positions = np.sort(fret_positions)

    if len(fret_positions) > max_frets + 1:
        fret_positions = fret_positions[:max_frets + 1]

    center_across = (fretboard_low + fretboard_high) / 2

    string_low = float(np.min(fret_positions))
    string_high = float(np.max(fret_positions))

    for pos in fret_positions:
        p_top = _point_from_projection(axis, perp, pos, fretboard_low)
        p_bot = _point_from_projection(axis, perp, pos, fretboard_high)

        p_top = tuple(np.round(p_top).astype(int))
        p_bot = tuple(np.round(p_bot).astype(int))

        cv2.line(output, p_top, p_bot, (255, 0, 255), 2)

    # Draw and label the six strings
    for i, across in enumerate(string_positions):
        p_left = _point_from_projection(axis, perp, string_low, across)
        p_right = _point_from_projection(axis, perp, string_high, across)

        p_left = tuple(np.round(p_left).astype(int))
        p_right = tuple(np.round(p_right).astype(int))

        cv2.line(output, p_left, p_right, (0, 255, 255), 2)

        label_point = _point_from_projection(axis, perp, string_low - 35, across)
        lx, ly = np.round(label_point).astype(int)
        cv2.putText(
            output,
            f"S{i + 1}",
            (lx - 10, ly + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

    for i in range(len(fret_positions) - 1):
        fret_number = i + 1
        label_pos_along = (fret_positions[i] + fret_positions[i + 1]) / 2
        label_point = _point_from_projection(axis, perp, label_pos_along, center_across)
        x, y = np.round(label_point).astype(int)

        cv2.putText(output,
            str(fret_number),
            (x - 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

    info = {
        "locked": False,
        "string_angle": string_angle,
        "fret_positions": fret_positions,
        "axis": axis,
        "perp": perp,
        "fretboard_low": fretboard_low,
        "fretboard_high": fretboard_high,
        "string_positions": string_positions,
        "string_low": string_low,
        "string_high": string_high,
    }

    global _last_gray, _last_info
    _last_gray = gray.copy()
    _last_info = info

    return output, info, edges


def _draw_locked_fretboard(output, tracked_points):
    fret_lines = []
    for i in range(0, len(tracked_points), 2):
        if i + 1 >= len(tracked_points):
            break

        p_top = tuple(np.round(tracked_points[i]).astype(int))
        p_bot = tuple(np.round(tracked_points[i + 1]).astype(int))
        fret_lines.append((p_top, p_bot))
        cv2.line(output, p_top, p_bot, (255, 0, 255), 2)

    for i in range(len(fret_lines) - 1):
        top1, bot1 = fret_lines[i]
        top2, bot2 = fret_lines[i + 1]

        x = int((top1[0] + bot1[0] + top2[0] + bot2[0]) / 4)
        y = int((top1[1] + bot1[1] + top2[1] + bot2[1]) / 4)

        cv2.putText(
            output,
            str(i + 1),
            (x - 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

    cv2.putText(
        output, "LOCKED",
        (16, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255, 0, 255), 2)


def _draw_locked_strings(output, tracked_strings):
    string_lines = []
    for i in range(0, len(tracked_strings), 2):
        if i + 1 >= len(tracked_strings):
            break

        p_left = tuple(np.round(tracked_strings[i]).astype(int))
        p_right = tuple(np.round(tracked_strings[i + 1]).astype(int))
        string_lines.append((p_left, p_right))
        cv2.line(output, p_left, p_right, (0, 255, 255), 2)

    for i, (p_left, p_right) in enumerate(string_lines):
        x = int(p_left[0] - 35)
        y = int(p_left[1])
        cv2.putText(output, f"S{i + 1}", (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)  # string labels


def detect_fretboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=15
    )

    output_frame = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output_frame, lines, edges


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
            drawn_frame, detected_lines, edge_mask = detect_fretboard_with_labels(frame)

            cv2.imshow("Fretboard Lines", drawn_frame)
            cv2.imshow("Canny Edges", edge_mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()