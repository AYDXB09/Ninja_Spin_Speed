import cv2
import numpy as np
import math
import time
from collections import deque

# ─────────────────────────────────────────
#  SETTINGS  (tweak if tracking is off)
# ─────────────────────────────────────────
WHITE_LOWER = np.array([0, 0, 220])    # HSV lower — tight white only
WHITE_UPPER = np.array([180, 25, 255]) # HSV upper — tight white only
MIN_AREA    = 20                        # ignore tiny blobs (noise)
MAX_AREA    = 500                       # ignore large blobs (background walls)
SMOOTHING   = 10                        # frames to average RPM over

# ─────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────
prev_angle   = None
prev_time    = None
rpm_history  = deque(maxlen=SMOOTHING)
pivot        = None

def find_white_tip(frame):
    """Find the centroid of the brightest small white blob (staff tip)."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best      = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA and area > best_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best      = (cx, cy)
                best_area = area

    return best, mask

def compute_rpm(tip, pivot):
    global prev_angle, prev_time

    dx    = tip[0] - pivot[0]
    dy    = tip[1] - pivot[1]
    angle = math.atan2(dy, dx)
    now   = time.time()

    rpm = 0
    if prev_angle is not None and prev_time is not None:
        dt = now - prev_time
        if dt > 0:
            delta = angle - prev_angle
            if delta >  math.pi: delta -= 2 * math.pi
            if delta < -math.pi: delta += 2 * math.pi
            rad_per_sec = delta / dt
            rpm = abs(rad_per_sec) * 60 / (2 * math.pi)

    prev_angle = angle
    prev_time  = now
    return rpm

def draw_overlay(frame, tip, pivot, rpm):
    h, w = frame.shape[:2]

    if tip and pivot:
        cv2.line(frame, pivot, tip, (0, 255, 0), 2)
        cv2.circle(frame, tip,   10, (0, 255, 255), -1)  # yellow = tip
        cv2.circle(frame, pivot,  6, (0, 0, 255),   -1)  # red    = pivot

    label = f"RPM: {rpm:.1f}"
    cv2.rectangle(frame, (10, 10), (320, 80), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    hint = "Click = set pivot  |  Q = quit"
    cv2.putText(frame, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame

def mouse_click(event, x, y, flags, param):
    global pivot
    if event == cv2.EVENT_LBUTTONDOWN:
        pivot = (x, y)
        print(f"✅ Pivot set to {pivot}")

def main():
    global rpm_history

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("SpinTracker")
    cv2.setMouseCallback("SpinTracker", mouse_click)

    # ── Trackbars to tune HSV live ──────────────────────────
    cv2.namedWindow("Tune HSV")
    cv2.createTrackbar("V min", "Tune HSV", 220, 255, lambda x: None)
    cv2.createTrackbar("S max", "Tune HSV",  25, 255, lambda x: None)
    cv2.createTrackbar("Area max","Tune HSV", 500, 3000, lambda x: None)

    print("✅ SpinTracker running!")
    print("   1. Hold toy in front of camera")
    print("   2. Use 'Tune HSV' sliders until ONLY the staff tip is white in the mask")
    print("   3. Click on the FIST to set pivot, then press the button!")
    print("   Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Read live slider values
        v_min    = cv2.getTrackbarPos("V min",    "Tune HSV")
        s_max    = cv2.getTrackbarPos("S max",    "Tune HSV")
        area_max = cv2.getTrackbarPos("Area max", "Tune HSV")

        WHITE_LOWER[2] = v_min
        WHITE_UPPER[1] = s_max
        MAX_AREA_live  = max(area_max, 50)

        # Find tip with live values
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tip      = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA_live and area > best_area:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    tip = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    best_area = area

        rpm = 0
        if tip and pivot:
            raw_rpm = compute_rpm(tip, pivot)
            rpm_history.append(raw_rpm)
            rpm = sum(rpm_history) / len(rpm_history)

        frame = draw_overlay(frame, tip, pivot, rpm)

        cv2.imshow("SpinTracker", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
