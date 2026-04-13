"""
Live (or video-file) spin rate tracker.

Recommended workflow:
- Put a bright tape marker on the staff tip.
- Tune HSV so only the marker is white in the mask.
- Set ROI around the marker area (prevents blob switching).
- Click pivot (rotation center).
- Read RPM live; export CSV for verification.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from collections import deque

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x0,y0,x1,y1


@dataclass
class Cfg:
    # HSV threshold
    h_min: int = 20
    h_max: int = 40
    s_min: int = 120
    s_max: int = 255
    v_min: int = 140
    v_max: int = 255

    min_area: int = 50
    max_area: int = 20000
    kernel: int = 5

    # Tracking gates
    max_tip_move_px: int = 120
    radius_consistency_px: int = 35

    # RPM
    rpm_window_s: float = 0.7
    max_rpm_plausible: float = 300.0


class PhaseMedianEstimator:
    def __init__(self, window_s: float, min_samples: int = 7):
        self.window_s = float(window_s)
        self.min_samples = int(min_samples)
        self._last_t: Optional[float] = None
        self._last_a: Optional[float] = None
        self._omegas: Deque[Tuple[float, float]] = deque()

    @staticmethod
    def wrap_to_pi(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def reset(self) -> None:
        self._last_t = None
        self._last_a = None
        self._omegas.clear()

    def update(self, t: float, a_wrapped: float, *, max_rad_s: Optional[float]) -> bool:
        t = float(t)
        a = float(a_wrapped)
        if self._last_t is None or self._last_a is None:
            self._last_t = t
            self._last_a = a
            return False
        if t <= self._last_t + 1e-12:
            return False
        dt = t - self._last_t
        dtheta = float(self.wrap_to_pi(a - self._last_a))
        omega = dtheta / max(dt, 1e-9)
        self._last_t = t
        self._last_a = a
        if max_rad_s is not None and abs(omega) > float(max_rad_s):
            return False
        self._omegas.append((t, omega))
        cutoff = t - self.window_s
        while self._omegas and self._omegas[0][0] < cutoff:
            self._omegas.popleft()
        return True

    def omega_med(self) -> float:
        if len(self._omegas) < self.min_samples:
            return 0.0
        vals = np.array([w for (_t, w) in self._omegas], dtype=np.float64)
        return float(np.median(vals))

    def rpm(self) -> float:
        return abs(self.omega_med()) * 60.0 / (2.0 * np.pi)


def normalize_rect(p0: Point, p1: Point) -> Rect:
    x0, y0 = p0
    x1, y1 = p1
    xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
    ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
    return int(xa), int(ya), int(xb), int(yb)


def apply_roi(mask: np.ndarray, roi: Optional[Rect]) -> np.ndarray:
    if roi is None:
        return mask
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = roi
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    out = np.zeros_like(mask)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def make_mask(frame: np.ndarray, cfg: Cfg) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg.h_min, cfg.s_min, cfg.v_min], dtype=np.uint8)
    upper = np.array([cfg.h_max, cfg.s_max, cfg.v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = max(1, int(cfg.kernel))
    if k % 2 == 0:
        k += 1
    ker = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    return mask


def find_tip(mask: np.ndarray, cfg: Cfg, prev_tip: Optional[Point], pivot: Optional[Point]) -> Optional[Point]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[Point] = None
    best_score = float("-inf")

    prev_r: Optional[float] = None
    if prev_tip is not None and pivot is not None:
        prx = float(prev_tip[0] - pivot[0])
        pry = float(prev_tip[1] - pivot[1])
        prev_r = float((prx * prx + pry * pry) ** 0.5)

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < cfg.min_area or area > cfg.max_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] <= 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        if pivot is None:
            cand = (cx, cy)
        else:
            pts = cnt.reshape(-1, 2)
            dxp = pts[:, 0].astype(np.float64) - float(pivot[0])
            dyp = pts[:, 1].astype(np.float64) - float(pivot[1])
            idx = int(np.argmax(dxp * dxp + dyp * dyp))
            cand = (int(pts[idx, 0]), int(pts[idx, 1]))

        if prev_tip is not None:
            dx = float(cand[0] - prev_tip[0])
            dy = float(cand[1] - prev_tip[1])
            dist = float((dx * dx + dy * dy) ** 0.5)
            if dist > float(cfg.max_tip_move_px):
                continue
            score = area - 0.03 * (dist * dist)
        else:
            score = area

        if pivot is not None and prev_r is not None:
            rx = float(cand[0] - pivot[0])
            ry = float(cand[1] - pivot[1])
            r = float((rx * rx + ry * ry) ** 0.5)
            if abs(r - prev_r) > float(cfg.radius_consistency_px):
                continue

        if score > best_score:
            best_score = score
            best = cand
    return best


def setup_trackbars(win: str, cfg: Cfg) -> None:
    cv2.namedWindow(win)
    cv2.createTrackbar("H min", win, cfg.h_min, 179, lambda _x: None)
    cv2.createTrackbar("H max", win, cfg.h_max, 179, lambda _x: None)
    cv2.createTrackbar("S min", win, cfg.s_min, 255, lambda _x: None)
    cv2.createTrackbar("S max", win, cfg.s_max, 255, lambda _x: None)
    cv2.createTrackbar("V min", win, cfg.v_min, 255, lambda _x: None)
    cv2.createTrackbar("V max", win, cfg.v_max, 255, lambda _x: None)
    cv2.createTrackbar("Area min", win, cfg.min_area, 5000, lambda _x: None)
    cv2.createTrackbar("Area max", win, cfg.max_area, 50000, lambda _x: None)
    cv2.createTrackbar("Kernel", win, cfg.kernel, 31, lambda _x: None)


def read_trackbars(win: str, cfg: Cfg) -> None:
    cfg.h_min = cv2.getTrackbarPos("H min", win)
    cfg.h_max = cv2.getTrackbarPos("H max", win)
    cfg.s_min = cv2.getTrackbarPos("S min", win)
    cfg.s_max = cv2.getTrackbarPos("S max", win)
    cfg.v_min = cv2.getTrackbarPos("V min", win)
    cfg.v_max = cv2.getTrackbarPos("V max", win)
    cfg.min_area = max(1, cv2.getTrackbarPos("Area min", win))
    cfg.max_area = max(cfg.min_area + 1, cv2.getTrackbarPos("Area max", win))
    cfg.kernel = max(1, cv2.getTrackbarPos("Kernel", win))
    if cfg.h_min > cfg.h_max:
        cfg.h_min, cfg.h_max = cfg.h_max, cfg.h_min
    if cfg.s_min > cfg.s_max:
        cfg.s_min, cfg.s_max = cfg.s_max, cfg.s_min
    if cfg.v_min > cfg.v_max:
        cfg.v_min, cfg.v_max = cfg.v_max, cfg.v_min


def save_hsv(path: str, key: str, cfg: Cfg) -> None:
    d = {
        "h_min": cfg.h_min, "h_max": cfg.h_max,
        "s_min": cfg.s_min, "s_max": cfg.s_max,
        "v_min": cfg.v_min, "v_max": cfg.v_max,
        "min_area": cfg.min_area, "max_area": cfg.max_area,
        "kernel": cfg.kernel,
    }
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)
            if isinstance(x, dict):
                data = x
    except FileNotFoundError:
        pass
    data[key] = d
    data["last"] = d
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_hsv(path: str, key: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    val = data.get(key) or data.get("last")
    return val if isinstance(val, dict) else None


def apply_hsv_dict(cfg: Cfg, d: dict) -> None:
    for k in ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max", "min_area", "max_area", "kernel"):
        if k in d:
            setattr(cfg, k, int(d[k]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="camera", help="camera or path to video")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--hsv-config", default=".spin_tracker_hsv.json")
    args = ap.parse_args()

    cfg = Cfg()
    is_camera = args.source.lower() == "camera"
    key = f"camera_{args.camera_index}" if is_camera else os.path.basename(args.source)
    loaded = load_hsv(args.hsv_config, key)
    if loaded:
        apply_hsv_dict(cfg, loaded)

    cap = cv2.VideoCapture(args.camera_index) if is_camera else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("❌ Cannot open source")
        return 2

    main_win = "Spin Tracker (Video | Mask)"
    tune_win = "Tune HSV"
    cv2.namedWindow(main_win)
    setup_trackbars(tune_win, cfg)

    pivot: Optional[Point] = None
    roi: Optional[Rect] = None
    roi_mode = False
    roi_p0: Optional[Point] = None
    prev_tip: Optional[Point] = None
    paused = not is_camera
    estimator = PhaseMedianEstimator(cfg.rpm_window_s)

    trace: List[Tuple[float, float, int, int, int, int, float, int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        nonlocal pivot, roi, roi_mode, roi_p0, prev_tip
        # ignore clicks on mask half
        if event == cv2.EVENT_LBUTTONDOWN:
            if frame_w > 0 and x >= frame_w:
                return
            if roi_mode:
                if roi_p0 is None:
                    roi_p0 = (x, y)
                    print("🟧 ROI start", roi_p0)
                else:
                    roi = normalize_rect(roi_p0, (x, y))
                    roi_mode = False
                    roi_p0 = None
                    prev_tip = None
                    estimator.reset()
                    print("🟧 ROI set", roi)
            else:
                pivot = (x, y)
                prev_tip = None
                estimator.reset()
                print("✅ Pivot set", pivot)

    cv2.setMouseCallback(main_win, on_mouse)

    fps = float(cap.get(cv2.CAP_PROP_FPS)) if not is_camera else 0.0
    if not is_camera and (not np.isfinite(fps) or fps <= 1e-3):
        fps = 30.0
    frame_period = 1.0 / fps if (not is_camera and not args.no_realtime) else 0.0
    next_t = time.perf_counter()

    last_frame = None
    frame_w = 0
    while True:
        if paused and last_frame is not None:
            frame = last_frame.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                if (not is_camera) and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    estimator.reset()
                    prev_tip = None
                    continue
                break
            last_frame = frame

        if not args.no_flip:
            frame = cv2.flip(frame, 1)
        frame_w = frame.shape[1]

        read_trackbars(tune_win, cfg)
        mask = apply_roi(make_mask(frame, cfg), roi)
        tip = find_tip(mask, cfg, prev_tip, pivot)

        t_s: Optional[float] = None
        if not paused and pivot is not None and tip is not None:
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) if not is_camera else float("nan")
            t_s = (pos_msec / 1000.0) if (not is_camera and np.isfinite(pos_msec) and pos_msec > 0) else time.perf_counter()
            dx = float(tip[0] - pivot[0])
            dy = float(tip[1] - pivot[1])
            ang = float(np.arctan2(dy, dx))
            max_rad_s = float(cfg.max_rpm_plausible) * 2.0 * np.pi / 60.0
            acc = 1 if estimator.update(t_s, ang, max_rad_s=max_rad_s) else 0
            rpm = estimator.rpm()
            trace.append((float(t_s), float(rpm), tip[0], tip[1], pivot[0], pivot[1], float(ang), int(cv2.countNonZero(mask)), acc))
            prev_tip = tip
        else:
            rpm = estimator.rpm()

        # overlay
        if roi is not None:
            x0, y0, x1, y1 = roi
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 120, 0), 2)
        if pivot is not None:
            cv2.circle(frame, pivot, 6, (0, 0, 255), -1)
        if pivot is not None and tip is not None:
            cv2.line(frame, pivot, tip, (0, 255, 0), 2)
            cv2.circle(frame, tip, 7, (0, 255, 255), -1)
        cv2.rectangle(frame, (10, 10), (330, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"RPM: {rpm:6.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "Click: pivot | O: ROI | C: clear ROI | Space: pause | W: save HSV | E: export CSV | Q: quit",
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        preview = np.hstack([frame, mask_bgr])
        cv2.imshow(main_win, preview)

        keycode = cv2.waitKey(1) & 0xFF
        if keycode in (ord("q"), ord("Q")):
            break
        if keycode == ord(" "):
            if not is_camera:
                paused = not paused
        if keycode in (ord("o"), ord("O")):
            roi_mode = True
            roi_p0 = None
            print("🟧 ROI mode: click two corners")
        if keycode in (ord("c"), ord("C")):
            roi = None
            roi_mode = False
            roi_p0 = None
            prev_tip = None
            estimator.reset()
            print("🟧 ROI cleared")
        if keycode in (ord("w"), ord("W")):
            save_hsv(args.hsv_config, key, cfg)
            print("💾 Saved HSV to", args.hsv_config, key)
        if keycode in (ord("e"), ord("E")):
            out = "rpm_trace.csv"
            with open(out, "w", encoding="utf-8") as f:
                f.write("t_s,rpm,tip_x,tip_y,pivot_x,pivot_y,angle_wrapped,mask_nonzero,accepted\n")
                for row in trace:
                    f.write(
                        f"{row[0]:.6f},{row[1]:.6f},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]:.6f},{row[7]},{row[8]}\n"
                    )
            print("📈 Wrote", out, len(trace), "rows")

        if frame_period > 0.0 and (not paused):
            next_t += frame_period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
