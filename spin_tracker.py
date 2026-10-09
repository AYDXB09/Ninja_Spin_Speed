"""
IB Physics HL — Rotational Mechanics Analysis App
==================================================
Real-time computer-vision tracker for a spinning staff/rod.

Tracks a bright colour marker on the tip of a spinning rod via webcam and
overlays live IB-style rotational kinematics & dynamics:

    θ (rad) | ω (rad s⁻¹) | α (rad s⁻²)
    v = ωr  | a_c = ω²r   | I, τ = Iα

Controls (shown on screen too):
    C       Calibrate: click the CENTER of rotation, then the marker TIP
    M       Sample marker colour: click on the marker in the (paused) frame
    V       Start/stop recording an annotated video (MP4)
    SPACE   Pause / resume the feed
    R       Reset angle/graph/data history
    S       Save recorded data to CSV (for your IA data analysis!)
    Q / ESC Quit

Run:    python spin_tracker.py
Video:  python spin_tracker.py --video clip.mp4      (press V first to also write an annotated MP4)
Test:   python spin_tracker.py --selftest   (validates physics math, no camera)
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

try:
    from scipy.signal import savgol_filter
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


# ----------------------------------------------------------------------------
# Text renderer — real Greek symbols (θ ω α τ) via PIL, cached as sprites
# ----------------------------------------------------------------------------
class TextRenderer:
    """Draws unicode text onto BGR frames. Sprites are cached by
    (text, size, color, bold) so per-frame cost is a simple alpha blit."""

    FONT_PATHS = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",   # macOS
        "/Library/Fonts/Arial Unicode.ttf",                       # macOS (older installs)
    ]
    BOLD_PATHS = [
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",   # macOS (no bold face; regular is used)
        "/Library/Fonts/Arial Unicode.ttf",
    ]

    def __init__(self):
        self.ok = HAVE_PIL
        self._fonts = {}
        self._cache = {}
        self._font_file = next((p for p in self.FONT_PATHS
                                if os.path.exists(p)), None)
        self._bold_file = next((p for p in self.BOLD_PATHS
                                if os.path.exists(p)), None)
        if self._font_file is None:
            self.ok = False

    def _font(self, size, bold):
        key = (size, bold)
        if key not in self._fonts:
            path = self._bold_file if (bold and self._bold_file) \
                else self._font_file
            self._fonts[key] = ImageFont.truetype(path, size)
        return self._fonts[key]

    def _sprite(self, text, size, color, bold):
        key = (text, size, color, bold)
        if key in self._cache:
            return self._cache[key]
        font = self._font(size, bold)
        l, t, r, b = font.getbbox(text)
        w, h = max(1, r - l), max(1, b - t)
        img = Image.new("RGBA", (w + 4, h + 4), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((2 - l, 2 - t), text, font=font,
                                 fill=(color[2], color[1], color[0], 255))
        arr = np.array(img)                     # RGBA
        sprite = (arr[:, :, [2, 1, 0]], arr[:, :, 3:4].astype(np.float32) / 255)
        if len(self._cache) > 600:              # keep the cache bounded
            self._cache.clear()
        self._cache[key] = sprite
        return sprite

    def put(self, img, text, org, size=16, color=(255, 255, 255), bold=False):
        """Draw `text` with its top-left at org. Falls back to cv2 text."""
        if not self.ok:
            cv2.putText(img, text.encode("ascii", "replace").decode(),
                        (int(org[0]), int(org[1]) + size),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 28, color, 1,
                        cv2.LINE_AA)
            return
        bgr, a = self._sprite(text, size, color, bold)
        h, w = bgr.shape[:2]
        x, y = int(org[0]), int(org[1])
        H, W = img.shape[:2]
        if x >= W or y >= H or x + w <= 0 or y + h <= 0:
            return
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        sx, sy = x0 - x, y0 - y
        roi = img[y0:y1, x0:x1].astype(np.float32)
        s_bgr = bgr[sy:sy + (y1 - y0), sx:sx + (x1 - x0)].astype(np.float32)
        s_a = a[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
        img[y0:y1, x0:x1] = (s_bgr * s_a + roi * (1 - s_a)).astype(np.uint8)

    def width(self, text, size=16, bold=False):
        if not self.ok:
            return int(len(text) * size * 0.55)
        font = self._font(size, bold)
        l, _, r, _ = font.getbbox(text)
        return r - l + 4


TEXT = TextRenderer()


# ----------------------------------------------------------------------------
# Physics engine
# ----------------------------------------------------------------------------
class PhysicsEngine:
    """Converts a stream of (t, x, y) marker positions into rotational metrics.

    Angle convention: standard physics (counter-clockwise positive), so the
    screen y-axis is flipped (screen y grows downward).
    The angle is unwrapped so theta accumulates over multiple revolutions.
    """

    def __init__(self, window=9):
        self.window = max(5, window | 1)          # odd, >= 5 (for Sav-Gol)
        self.t = deque(maxlen=600)                # ~20 s at 30 fps
        self.theta = deque(maxlen=600)            # unwrapped, rad
        self.omega_hist = deque(maxlen=600)
        self.alpha_hist = deque(maxlen=600)
        self._last_raw = None                     # last wrapped angle

    def reset(self):
        self.t.clear(); self.theta.clear()
        self.omega_hist.clear(); self.alpha_hist.clear()
        self._last_raw = None

    def add(self, t, x, y, xc, yc):
        """Add a marker sample. Returns (theta, omega, alpha)."""
        raw = math.atan2(-(y - yc), x - xc)       # flip y: CCW positive
        if self._last_raw is None:
            unwrapped = raw
        else:
            d = raw - self._last_raw
            # unwrap: assume < half a turn between frames
            if d > math.pi:
                d -= 2 * math.pi
            elif d < -math.pi:
                d += 2 * math.pi
            unwrapped = self.theta[-1] + d
        self._last_raw = raw
        self.t.append(t)
        self.theta.append(unwrapped)
        omega, alpha = self._derivatives()
        self.omega_hist.append(omega)
        self.alpha_hist.append(alpha)
        return unwrapped, omega, alpha

    def predict_angle(self, t):
        """Extrapolate the unwrapped angle to time t using current omega."""
        if len(self.t) < 3:
            return None
        return self.theta[-1] + self.omega_hist[-1] * (t - self.t[-1])

    def _derivatives(self):
        """omega, alpha from a smoothed window of recent theta samples."""
        n = min(len(self.theta), self.window)
        if n < 3:
            return 0.0, 0.0
        ts = np.array(list(self.t)[-n:])
        th = np.array(list(self.theta)[-n:])
        dt = (ts[-1] - ts[0]) / (n - 1)
        if dt <= 0:
            return 0.0, 0.0
        if HAVE_SCIPY and n >= 5:
            # Savitzky-Golay: simultaneous smoothing + differentiation
            omega = savgol_filter(th, n, polyorder=2, deriv=1, delta=dt)[-2]
            alpha = savgol_filter(th, n, polyorder=2, deriv=2, delta=dt)[-2]
        else:
            omega = (th[-1] - th[-3]) / (2 * dt)              # central difference
            alpha = (th[-1] - 2 * th[-2] + th[-3]) / (dt * dt)  # second difference
        return float(omega), float(alpha)


# ----------------------------------------------------------------------------
# Marker tracker — HSV colour + motion mask + circle constraint + prediction
# ----------------------------------------------------------------------------
class ColorTracker:
    """Finds the tip marker robustly, even at high spin rates.

    Layered defences against both failure modes seen in practice:

    * Wrong-object jumps (grabbing a similar colour in the background or on
      the toy's body): once calibrated, the search is restricted to a thin
      ANNULUS around the spin circle — the tip physically cannot be anywhere
      else — and candidates far from the PREDICTED position (extrapolated
      from the current omega) are rejected.

    * Losing the marker at speed: motion blur desaturates the tape so the
      strict colour range misses it. A RELAXED colour range is also checked,
      but only where the MOTION MASK (frame differencing) says something is
      actually moving — so the relaxed range cannot latch onto static
      background objects.
    """

    def __init__(self):
        # default: bright yellow tape
        self.lower = np.array([20, 90, 90])
        self.upper = np.array([38, 255, 255])
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.last = None          # last accepted (x, y)
        self.lost = 0             # consecutive frames with no accepted blob
        self._relax()

    def _relax(self):
        """Relaxed range: same hue band, much lower sat/val floors (blur)."""
        self.lower_rx = self.lower.copy()
        self.lower_rx[1] = max(25, int(self.lower[1] * 0.45))
        self.lower_rx[2] = max(40, int(self.lower[2] * 0.5))
        self.upper_rx = self.upper.copy()

    def reset(self):
        self.last = None
        self.lost = 0

    def sample(self, frame_bgr, x, y):
        """Re-tune the HSV range from a clicked pixel (5x5 median)."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        y0, y1 = max(0, y - 2), y + 3
        x0, x1 = max(0, x - 2), x + 3
        patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
        h, s, v = np.median(patch, axis=0)
        self.lower = np.array([max(0, h - 12), max(50, s - 70), max(50, v - 80)])
        self.upper = np.array([min(179, h + 12), 255, 255])
        self._relax()
        self.reset()
        return (h, s, v)

    def find(self, frame_bgr, center=None, r_px=None,
             predict=None, motion=None):
        """Locate the marker.

        center/r_px : calibration -> annulus constraint
        predict     : (x, y) expected from physics -> fast-motion gating
        motion      : uint8 mask of moving pixels -> enables relaxed colours
        Returns ((x, y), debug_mask) or (None, debug_mask).
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        if motion is not None:
            relaxed = cv2.inRange(hsv, self.lower_rx, self.upper_rx)
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(relaxed, motion))

        # the tip can only ever be on the spin circle: annulus restriction
        if center is not None and r_px:
            ann = np.zeros(mask.shape, np.uint8)
            band = max(int(0.30 * r_px), 26)
            cv2.circle(ann, (int(center[0]), int(center[1])),
                       int(r_px + band), 255, -1)
            cv2.circle(ann, (int(center[0]), int(center[1])),
                       max(1, int(r_px - band)), 0, -1)
            mask = cv2.bitwise_and(mask, ann)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < 12:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cands.append((M["m10"] / M["m00"], M["m01"] / M["m00"], a))
        if not cands:
            self.lost += 1
            if self.lost > 8:
                self.last = None
            return None, mask

        # gate distance: around the physics prediction if available (this is
        # what lets a FAST marker that moved 150 px in one frame still pass),
        # else around the last seen position, widening while lost
        anchor = predict if predict is not None else self.last
        if predict is not None and r_px:
            gate = max(0.9 * r_px, 110)
        elif r_px:
            # calibrated but no prediction yet (bootstrap): the annulus is
            # already constraining the search, so allow up to a half-turn
            # per frame rather than rejecting genuinely fast markers
            gate = 2.1 * r_px
        else:
            gate = 90 + 55 * min(self.lost, 6)

        best, best_score = None, -1e18
        for (x, y, a) in cands:
            score = 0.30 * min(a, 400)                 # mild size preference
            if anchor is not None:
                d = math.hypot(x - anchor[0], y - anchor[1])
                if d > gate:
                    continue
                score -= 1.2 * d
            if center is not None and r_px:
                dev = abs(math.hypot(x - center[0], y - center[1]) - r_px)
                score -= 1.0 * dev                     # closer to circle better
            if score > best_score:
                best_score, best = score, (x, y)

        if best is None:                               # all candidates gated out
            self.lost += 1
            if self.lost > 8:
                self.last = None
            return None, mask
        self.last = best
        self.lost = 0
        return best, mask


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def draw_arrow(img, origin, vec, color, scale=1.0, min_len=20):
    """Bold, outlined vector arrow. Returns the tip point (or None)."""
    ox, oy = int(round(origin[0])), int(round(origin[1]))
    ex = origin[0] + vec[0] * scale
    ey = origin[1] + vec[1] * scale
    if math.hypot(ex - ox, ey - oy) < min_len:
        return None
    tip = (int(round(ex)), int(round(ey)))
    cv2.arrowedLine(img, (ox, oy), tip, (0, 0, 0), 7, cv2.LINE_AA,
                    tipLength=0.22)                    # dark outline
    cv2.arrowedLine(img, (ox, oy), tip, color, 3, cv2.LINE_AA, tipLength=0.22)
    return tip


def draw_vector_label(img, text, anchor, color, size=15):
    """Label with dark backdrop so it reads over any background."""
    w = TEXT.width(text, size, bold=True)
    x, y = int(anchor[0]), int(anchor[1])
    x = min(max(4, x), img.shape[1] - w - 6)
    y = min(max(4, y), img.shape[0] - size - 10)
    overlay = img[y - 3:y + size + 7, x - 4:x + w + 4]
    if overlay.size:
        cv2.rectangle(img, (x - 4, y - 3), (x + w + 4, y + size + 7),
                      (10, 10, 10), -1)
    TEXT.put(img, text, (x, y), size, color, bold=True)


def draw_spin_arrow(img, center, radius, omega, color):
    """Curved arrow around the axis showing the sense of rotation (CW/CCW)."""
    if abs(omega) < 0.3:
        return
    cx, cy = int(center[0]), int(center[1])
    ccw = omega > 0
    # arc from -60 deg to +200 deg sweep, drawn as an ellipse arc
    start, end = (200, 340) if ccw else (340, 200)
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, start, end,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, start, end,
                color, 2, cv2.LINE_AA)
    # arrow head at the arc end
    a = math.radians(end)
    hx = cx + radius * math.cos(a)
    hy = cy + radius * math.sin(a)
    # tangent direction at end point (screen coords)
    tx, ty = -math.sin(a), math.cos(a)
    if not ccw:
        tx, ty = -tx, -ty
    p1 = (int(hx + 10 * tx), int(hy + 10 * ty))
    cv2.arrowedLine(img, (int(hx - 2 * tx), int(hy - 2 * ty)), p1,
                    (0, 0, 0), 5, cv2.LINE_AA, tipLength=1.8)
    cv2.arrowedLine(img, (int(hx - 2 * tx), int(hy - 2 * ty)), p1,
                    color, 2, cv2.LINE_AA, tipLength=1.8)


def fmt_num(val, dec=2):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    a = abs(val)
    if a != 0 and (a >= 10000 or a < 0.01):
        return f"{val:.2e}"
    return f"{val:,.{dec}f}"


def draw_graph(panel, xs, ys, color, title, unit, t_span=10.0):
    """Rolling line plot with axes, drawn straight onto an OpenCV panel."""
    h, w = panel.shape[:2]
    ml, mr, mt, mb = 64, 14, 30, 22                     # margins
    gw, gh = w - ml - mr, h - mt - mb
    # plot area
    cv2.rectangle(panel, (ml, mt), (ml + gw, mt + gh), (52, 48, 44), -1)
    cv2.rectangle(panel, (ml, mt), (ml + gw, mt + gh), (86, 82, 76), 1)
    TEXT.put(panel, title, (ml, 6), 14, (225, 225, 225), bold=True)
    TEXT.put(panel, unit, (ml + TEXT.width(title, 14, True) + 8, 7), 12,
             (150, 150, 150))
    if len(xs) < 2:
        TEXT.put(panel, "waiting for data…", (ml + gw // 2 - 50,
                 mt + gh // 2 - 8), 13, (120, 120, 120))
        return
    t_end = xs[-1]
    t_start = t_end - t_span
    pts_t, pts_y = [], []
    for t, y in zip(xs, ys):
        if t >= t_start:
            pts_t.append(t); pts_y.append(y)
    if len(pts_t) < 2:
        return
    y_max = max(1e-6, max(abs(min(pts_y)), abs(max(pts_y)))) * 1.15
    y_zero = mt + gh // 2
    # gridlines + labels
    for frac, yv in ((0.0, y_max), (0.5, 0.0), (1.0, -y_max)):
        gy = int(mt + frac * gh)
        cv2.line(panel, (ml, gy), (ml + gw, gy), (66, 62, 58), 1)
        lbl = fmt_num(yv, 1)
        TEXT.put(panel, lbl, (ml - TEXT.width(lbl, 11) - 5, gy - 7), 11,
                 (165, 165, 165))
    cv2.line(panel, (ml, y_zero), (ml + gw, y_zero), (95, 90, 84), 1)
    poly = []
    for t, y in zip(pts_t, pts_y):
        px = ml + int((t - t_start) / t_span * gw)
        py = int(mt + gh / 2 - (y / y_max) * (gh / 2))
        poly.append((px, py))
    cv2.polylines(panel, [np.array(poly, np.int32)], False, color, 2,
                  cv2.LINE_AA)
    # current-value dot + time axis label
    cv2.circle(panel, poly[-1], 4, color, -1, cv2.LINE_AA)
    TEXT.put(panel, f"t (last {t_span:.0f} s)", (ml + gw - 78, mt + gh + 4),
             12, (150, 150, 150))


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
class App:
    STATE_RUN = "run"
    STATE_CLICK_CENTER = "click_center"
    STATE_CLICK_TIP = "click_tip"
    STATE_CLICK_COLOR = "click_color"

    # colour theme (BGR)
    C_BG = (30, 27, 24)
    C_PANEL = (42, 38, 34)
    C_CARD = (52, 48, 43)
    C_EDGE = (86, 80, 72)
    C_HEAD = (255, 214, 120)       # headings (soft cyan-gold)
    C_VALUE = (250, 250, 250)
    C_DIM = (170, 170, 170)
    C_V = (90, 225, 90)            # velocity — green
    C_AC = (60, 150, 255)          # centripetal accel — orange
    C_OMEGA = (230, 140, 255)      # spin direction — violet
    C_ANCHOR = (70, 70, 255)       # axis — red

    def __init__(self, source, length_m, mass_kg, pivot):
        self.tracker = ColorTracker()
        self.engine = PhysicsEngine()
        self.length_m = length_m
        self.mass_kg = mass_kg
        self.pivot = pivot                        # "center" or "end"
        if pivot == "center":
            self.r_m = length_m / 2.0
            self.I = mass_kg * length_m ** 2 / 12.0
        else:
            self.r_m = length_m
            self.I = mass_kg * length_m ** 2 / 3.0
        self.source = source                      # int camera or str filename
        self.center = None                        # (x, y) px
        self.r_px = None
        self.state = self.STATE_RUN
        self.paused = False
        self.frozen = None
        self.msg = "Press C to calibrate: click the axle, then the marker tip"
        self.records = []                         # rows for CSV export
        self.t0 = None
        self.tip = None
        self.prev_gray = None
        self.writer = None                        # annotated video writer
        self.export_path = None                   # auto-export (video mode)

    # ---- mouse -------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.state == self.STATE_CLICK_CENTER:
            self.center = (float(x), float(y))
            self.state = self.STATE_CLICK_TIP
            self.msg = "Step 2/2 — now click the coloured MARKER on the tip"
        elif self.state == self.STATE_CLICK_TIP:
            if self.center:
                self.r_px = math.hypot(x - self.center[0], y - self.center[1])
                if self.r_px < 5:
                    self.msg = "Too close to the centre — click the tip again"
                    return
            self.state = self.STATE_RUN
            self.paused = False
            self.engine.reset()
            self.tracker.reset()
            self.msg = (f"Calibrated ✓   r = {self.r_m * 100:.1f} cm = "
                        f"{self.r_px:.0f} px   "
                        f"({self.r_m / self.r_px * 1000:.2f} mm per pixel)")
        elif self.state == self.STATE_CLICK_COLOR:
            src = self.frozen if self.frozen is not None else None
            if src is not None and y < src.shape[0] and x < src.shape[1]:
                h, s, v = self.tracker.sample(src, x, y)
                self.msg = f"Marker colour locked ✓  (hue {h:.0f})"
            self.state = self.STATE_RUN
            self.paused = False

    # ---- main loop ---------------------------------------------------------
    def run(self):
        is_file = isinstance(self.source, str)
        if is_file:
            cap = cv2.VideoCapture(self.source)
        else:
            # DirectShow is Windows-only; other platforms use OpenCV's default backend
            cap = (cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                   if sys.platform == "win32" else cv2.VideoCapture(self.source))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 60)
        if not cap.isOpened():
            print(f"ERROR: cannot open source {self.source}. "
                  "Try --camera 1 (or 2).")
            return 1
        file_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else None
        if file_fps and not (1 <= file_fps <= 240):
            file_fps = 30.0
        frame_i = 0

        win = "IB Rotational Mechanics Tracker"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)
        self.t0 = time.perf_counter()
        if is_file:
            self.paused = True
            self.msg = ("Video loaded — press M (lock colour) and C (calibrate) "
                        "on this frame; analysis starts automatically")
            ok, first = cap.read()
            if ok:
                self.frozen = first

        while True:
            if self.paused and self.frozen is not None:
                frame = self.frozen.copy()
                advanced = False
            else:
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        self.msg = "End of video — S saves CSV, Q quits"
                        self.paused = True
                        frame = self.frozen.copy() if self.frozen is not None \
                            else np.zeros((480, 640, 3), np.uint8)
                        advanced = False
                    else:
                        self.msg = "Camera frame dropped…"
                        if cv2.waitKey(20) in (27, ord('q')):
                            break
                        continue
                else:
                    self.frozen = frame.copy()
                    advanced = True
                    frame_i += 1
            # timebase: real clock for live camera, frame count for files
            if is_file:
                t = frame_i / file_fps
            else:
                t = time.perf_counter() - self.t0

            theta = omega = alpha = v = a_c = tau = float("nan")
            if advanced and self.state == self.STATE_RUN:
                # motion mask from frame differencing (catches blurred marker)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = None
                if self.prev_gray is not None:
                    diff = cv2.absdiff(gray, self.prev_gray)
                    _, motion = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
                    motion = cv2.dilate(motion, None, iterations=3)
                self.prev_gray = gray
                # physics-based position prediction for fast motion
                predict = None
                if self.center and self.r_px:
                    th_p = self.engine.predict_angle(t)
                    if th_p is not None:
                        predict = (self.center[0] + self.r_px * math.cos(th_p),
                                   self.center[1] - self.r_px * math.sin(th_p))
                tip, _mask = self.tracker.find(frame, self.center, self.r_px,
                                               predict, motion)
                self.tip = tip
                if tip and self.center and self.r_px:
                    theta, omega, alpha = self.engine.add(
                        t, tip[0], tip[1], self.center[0], self.center[1])
                    v = omega * self.r_m
                    a_c = omega ** 2 * self.r_m
                    tau = self.I * alpha
                    self.records.append((t, theta, omega, alpha, v, a_c, tau))
            elif self.engine.theta:
                theta = self.engine.theta[-1]
                omega = self.engine.omega_hist[-1]
                alpha = self.engine.alpha_hist[-1]
                v = omega * self.r_m
                a_c = omega ** 2 * self.r_m
                tau = self.I * alpha

            canvas = self.compose(frame, t, theta, omega, alpha, v, a_c, tau)
            if self.writer is not None and advanced:
                self.writer.write(canvas)
            cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('c'):
                self.paused = True
                self.state = self.STATE_CLICK_CENTER
                self.msg = "Step 1/2 — click the CENTRE of rotation (the axle)"
            elif key == ord('m'):
                self.paused = True
                self.state = self.STATE_CLICK_COLOR
                self.msg = "Click directly ON the colour marker"
            elif key == ord(' '):
                self.paused = not self.paused
                self.msg = "Paused" if self.paused else "Running"
            elif key == ord('r'):
                self.engine.reset()
                self.tracker.reset()
                self.records.clear()
                self.t0 = time.perf_counter()
                frame_i = 0
                self.msg = "Reset — angle, graphs and data cleared"
            elif key == ord('s'):
                self.save_csv()
            elif key == ord('v'):
                self.toggle_recording(canvas.shape)

        if self.writer is not None:
            self.writer.release()
            print(f"Annotated video saved -> {self.export_path}")
        if is_file and self.records:
            self.save_csv()
        cap.release()
        cv2.destroyAllWindows()
        return 0

    def toggle_recording(self, shape):
        if self.writer is None:
            self.export_path = time.strftime("spin_annotated_%Y%m%d_%H%M%S.mp4")
            h, w = shape[:2]
            self.writer = cv2.VideoWriter(
                self.export_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                (w, h))
            self.msg = f"● Recording annotated video -> {self.export_path}"
        else:
            self.writer.release()
            self.writer = None
            self.msg = f"Recording saved ✓  {self.export_path}"
            print(self.msg)

    # ---- rendering ---------------------------------------------------------
    def compose(self, frame, t, theta, omega, alpha, v, a_c, tau):
        fh, fw = frame.shape[:2]
        panel_h, graph_h = 148, 185
        canvas = np.full((fh + panel_h + graph_h, fw, 3), self.C_BG, np.uint8)

        vid = frame
        self._overlay_physics(vid, theta, omega, v, a_c)
        canvas[:fh, :fw] = vid
        self._header(canvas, fw)
        self._footer(canvas, fw, fh)
        self._metric_cards(canvas, fw, fh, panel_h,
                           theta, omega, alpha, v, a_c, tau)
        self._graphs(canvas, fw, fh, panel_h, graph_h)
        return canvas

    def _overlay_physics(self, vid, theta, omega, v, a_c):
        if self.center:
            cx, cy = int(self.center[0]), int(self.center[1])
            if self.r_px:
                # spin circle (the tip's path)
                cv2.circle(vid, (cx, cy), int(self.r_px), (130, 130, 130), 1,
                           cv2.LINE_AA)
                # reference axis for theta (dashed, along +x)
                for d in range(0, int(self.r_px), 14):
                    cv2.line(vid, (cx + d, cy), (cx + min(d + 7,
                             int(self.r_px)), cy), (110, 110, 110), 1,
                             cv2.LINE_AA)
                # curved arrow showing sense of rotation
                if not math.isnan(omega):
                    draw_spin_arrow(vid, self.center,
                                    max(18, int(self.r_px * 0.22)),
                                    omega, self.C_OMEGA)
            # axis anchor
            cv2.circle(vid, (cx, cy), 7, self.C_ANCHOR, -1, cv2.LINE_AA)
            cv2.circle(vid, (cx, cy), 11, (255, 255, 255), 1, cv2.LINE_AA)

        if self.tip and self.center:
            tx, ty = int(self.tip[0]), int(self.tip[1])
            cx, cy = int(self.center[0]), int(self.center[1])
            rx, ry = self.tip[0] - self.center[0], self.tip[1] - self.center[1]
            rlen = math.hypot(rx, ry)
            # angle arc from reference axis to the radius arm
            if self.r_px and not math.isnan(theta) and rlen > 1:
                arc_r = max(14, int(self.r_px * 0.4))
                ang = -math.degrees(math.atan2(-ry, rx))   # screen angle
                sweep = ang % 360
                cv2.ellipse(vid, (cx, cy), (arc_r, arc_r), 0, 0, ang,
                            (200, 200, 90), 2, cv2.LINE_AA)
                mid = math.radians(ang / 2)
                TEXT.put(vid, "θ",
                         (int(cx + (arc_r + 12) * math.cos(mid)) - 5,
                          int(cy + (arc_r + 12) * math.sin(mid)) - 9),
                         17, (210, 210, 120), bold=True)
                _ = sweep
            # radius arm
            cv2.line(vid, (cx, cy), (tx, ty), (240, 240, 240), 2, cv2.LINE_AA)
            # tip marker
            cv2.circle(vid, (tx, ty), 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vid, (tx, ty), 7, (60, 255, 255), -1, cv2.LINE_AA)
            if rlen > 1 and not math.isnan(omega):
                ux, uy = rx / rlen, ry / rlen
                txv, tyv = uy, -ux            # tangent, CCW-positive on screen
                sgn = 1 if omega >= 0 else -1
                vscale = float(np.clip(abs(v) * 75, 0, 160))
                tip_pt = draw_arrow(vid, (tx, ty), (sgn * txv, sgn * tyv),
                                    self.C_V, scale=vscale)
                if tip_pt:
                    draw_vector_label(vid, f"v = {abs(v):.2f} m s⁻¹",
                                      (tip_pt[0] + 6, tip_pt[1] - 20),
                                      self.C_V)
                ascale = float(np.clip(abs(a_c) * 2.5, 0, 140))
                tip_pt = draw_arrow(vid, (tx, ty), (-ux, -uy), self.C_AC,
                                    scale=ascale)
                if tip_pt:
                    draw_vector_label(vid, f"a꜀ = {abs(a_c):.1f} m s⁻²",
                                      (tip_pt[0] + 6, tip_pt[1] + 6),
                                      self.C_AC)
        elif self.tip:                        # tracked but not yet calibrated
            tx, ty = int(self.tip[0]), int(self.tip[1])
            cv2.circle(vid, (tx, ty), 7, (60, 255, 255), -1, cv2.LINE_AA)

    def _header(self, canvas, fw):
        strip = canvas[0:36, 0:fw].copy()
        cv2.rectangle(strip, (0, 0), (fw, 36), (0, 0, 0), -1)
        cv2.addWeighted(strip, 0.6, canvas[0:36, 0:fw], 0.4, 0,
                        canvas[0:36, 0:fw])
        TEXT.put(canvas, "Rotational Mechanics Lab", (12, 7), 18,
                 (255, 255, 255), bold=True)
        TEXT.put(canvas, "IB Physics HL · A.4",
                 (16 + TEXT.width("Rotational Mechanics Lab", 18, True), 11),
                 13, (185, 185, 185))
        # status pill
        if not self.center or not self.r_px:
            txt, col = "CALIBRATE (press C)", (60, 200, 255)
        elif self.writer is not None:
            txt, col = "● REC + TRACKING", (80, 80, 255)
        elif self.tracker.lost == 0 and self.tip:
            txt, col = "TRACKING LOCKED", (90, 235, 90)
        else:
            txt, col = "SEARCHING…", (60, 200, 255)
        w = TEXT.width(txt, 13, True)
        x = fw - w - 30
        cv2.circle(canvas, (x - 8, 18), 5, col, -1, cv2.LINE_AA)
        TEXT.put(canvas, txt, (x, 10), 13, col, bold=True)

    def _footer(self, canvas, fw, fh):
        strip = canvas[fh - 48:fh, 0:fw].copy()
        cv2.rectangle(strip, (0, 0), (fw, 48), (0, 0, 0), -1)
        cv2.addWeighted(strip, 0.55, canvas[fh - 48:fh, 0:fw], 0.45, 0,
                        canvas[fh - 48:fh, 0:fw])
        TEXT.put(canvas, self.msg, (10, fh - 46), 14, (120, 225, 255))
        TEXT.put(canvas,
                 "C calibrate    M marker colour    SPACE pause    R reset"
                 "    S save CSV    V record video    Q quit",
                 (10, fh - 24), 13, (200, 200, 200))

    # ---- metric cards ------------------------------------------------------
    def _card(self, canvas, x, y, w, h, accent, sym, name, val, unit,
              formula, extra=""):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.C_CARD, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.C_EDGE, 1)
        cv2.rectangle(canvas, (x, y), (x + 3, y + h), accent, -1)
        TEXT.put(canvas, sym, (x + 10, y + 4), 19, accent, bold=True)
        TEXT.put(canvas, name, (x + 10 + TEXT.width(sym, 19, True) + 6, y + 8),
                 12, self.C_DIM)
        TEXT.put(canvas, val, (x + 10, y + 26), 20, self.C_VALUE, bold=True)
        TEXT.put(canvas, unit, (x + 12 + TEXT.width(val, 20, True), y + 32),
                 12, self.C_DIM)
        TEXT.put(canvas, formula, (x + 10, y + h - 18), 12, (150, 190, 230))
        if extra:
            TEXT.put(canvas, extra,
                     (x + w - TEXT.width(extra, 12) - 8, y + h - 18), 12,
                     (150, 200, 150))

    def _metric_cards(self, canvas, fw, fh, panel_h,
                      theta, omega, alpha, v, a_c, tau):
        y0 = fh
        cv2.rectangle(canvas, (0, y0), (fw, y0 + panel_h), self.C_PANEL, -1)
        cv2.line(canvas, (0, y0), (fw, y0), self.C_EDGE, 1)
        f_hz = abs(omega) / (2 * math.pi) if not math.isnan(omega) else float("nan")
        revs = theta / (2 * math.pi) if not math.isnan(theta) else float("nan")

        gap = 8
        cols = 3
        cw = (fw - gap * (cols + 1)) // cols
        ch = (panel_h - gap * 3) // 2
        cards = [
            (self.C_HEAD, "θ", "angular displacement", fmt_num(theta),
             "rad", "θ = s / r",
             "" if math.isnan(revs) else f"{revs:+.2f} rev"),
            (self.C_OMEGA, "ω", "angular velocity", fmt_num(omega),
             "rad s⁻¹", "ω = Δθ / Δt",
             "" if math.isnan(f_hz) else f"{f_hz:.2f} Hz"),
            ((160, 160, 255), "α", "angular acceleration", fmt_num(alpha),
             "rad s⁻²", "α = Δω / Δt", ""),
            (self.C_V, "v", "tangential speed", fmt_num(abs(v))
             if not math.isnan(v) else "—",
             "m s⁻¹", f"v = ωr   (r = {self.r_m * 100:.1f} cm)", ""),
            (self.C_AC, "a꜀", "centripetal acceleration", fmt_num(a_c),
             "m s⁻²", "a꜀ = ω²r  → centre", ""),
            ((120, 220, 255), "τ", "net torque", fmt_num(tau, 4),
             "N m", f"τ = Iα · I = {self.I:.1e} kg m²", ""),
        ]
        for i, (accent, sym, name, val, unit, formula, extra) in enumerate(cards):
            r, c = divmod(i, cols)
            x = gap + c * (cw + gap)
            y = y0 + gap + r * (ch + gap)
            self._card(canvas, x, y, cw, ch, accent, sym, name, val, unit,
                       formula, extra)

    def _graphs(self, canvas, fw, fh, panel_h, graph_h):
        gy = fh + panel_h
        cv2.line(canvas, (0, gy), (fw, gy), self.C_EDGE, 1)
        half = fw // 2
        ts = list(self.engine.t)
        draw_graph(canvas[gy:gy + graph_h, :half], ts,
                   list(self.engine.omega_hist), self.C_V,
                   "ω – t", "rad s⁻¹")
        draw_graph(canvas[gy:gy + graph_h, half:], ts,
                   [self.I * a for a in self.engine.alpha_hist], self.C_AC,
                   "τ – t", "N m")

    # ---- data export -------------------------------------------------------
    def save_csv(self):
        if not self.records:
            self.msg = "No data recorded yet — calibrate and spin first"
            return
        name = time.strftime("spin_data_%Y%m%d_%H%M%S.csv")
        with open(name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t (s)", "theta (rad)", "omega (rad/s)",
                        "alpha (rad/s^2)", "v (m/s)", "a_c (m/s^2)",
                        "tau (N m)"])
            w.writerows(self.records)
        self.msg = f"Saved {len(self.records)} samples -> {name}"
        print(self.msg)


# ----------------------------------------------------------------------------
# Self-test: feed synthetic rotation through the pipeline, check accuracy
# ----------------------------------------------------------------------------
def selftest():
    print("Self-test: synthetic rotation through the physics engine")
    ok = True

    # Test 1: constant omega = 10 rad/s
    eng = PhysicsEngine()
    fps, w_true, r = 60.0, 10.0, 200.0
    for i in range(240):
        t = i / fps
        th = w_true * t
        x = 480 + r * math.cos(th)
        y = 270 - r * math.sin(th)          # screen y down => CCW on screen
        theta, omega, alpha = eng.add(t, x, y, 480, 270)
    err_w = abs(omega - w_true)
    err_th = abs(theta - w_true * t)
    print(f"  [1] const spin : omega={omega:.4f} (true 10), "
          f"theta={theta:.3f} (true {w_true * t:.3f}), alpha={alpha:.4f} (true 0)")
    ok &= err_w < 0.05 and err_th < 0.01 and abs(alpha) < 0.5

    # Test 2: constant angular acceleration alpha = 5 rad/s^2
    eng = PhysicsEngine()
    a_true = 5.0
    for i in range(240):
        t = i / fps
        th = 0.5 * a_true * t * t
        x = 480 + r * math.cos(th)
        y = 270 - r * math.sin(th)
        theta, omega, alpha = eng.add(t, x, y, 480, 270)
    w_expect = a_true * t
    print(f"  [2] accel spin : omega={omega:.4f} (true {w_expect:.3f}), "
          f"alpha={alpha:.4f} (true 5)")
    ok &= abs(omega - w_expect) < 0.1 and abs(alpha - a_true) < 0.3

    # Test 3: clockwise spin gives negative omega
    eng = PhysicsEngine()
    for i in range(120):
        t = i / fps
        th = -8.0 * t
        x = 480 + r * math.cos(th)
        y = 270 - r * math.sin(th)
        theta, omega, alpha = eng.add(t, x, y, 480, 270)
    print(f"  [3] CW spin    : omega={omega:.4f} (true -8)")
    ok &= abs(omega + 8.0) < 0.05

    # Test 4: noisy pixels (+/-2 px jitter), omega stays within 3%
    eng = PhysicsEngine()
    rng = np.random.default_rng(42)
    for i in range(300):
        t = i / fps
        th = 12.0 * t
        x = 480 + r * math.cos(th) + rng.normal(0, 2)
        y = 270 - r * math.sin(th) + rng.normal(0, 2)
        theta, omega, alpha = eng.add(t, x, y, 480, 270)
    ws = np.array(list(eng.omega_hist)[60:])
    mean_w, sd_w = ws.mean(), ws.std()
    print(f"  [4] noisy spin : mean omega={mean_w:.3f} (true 12), sd={sd_w:.3f}")
    ok &= abs(mean_w - 12.0) < 0.36

    # Test 5: HSV tracker finds a synthetic yellow dot
    frame = np.zeros((540, 960, 3), np.uint8)
    cv2.circle(frame, (600, 200), 10, (0, 230, 240), -1)   # BGR yellow
    tip, _ = ColorTracker().find(frame)
    found = tip is not None and abs(tip[0] - 600) < 2 and abs(tip[1] - 200) < 2
    print(f"  [5] HSV tracker: found={found} at {tip}")
    ok &= found

    # Test 6: gating rejects a big background blob far off the spin circle
    center, r_px = (480, 270), 200.0
    trk = ColorTracker()
    real = (480 + int(r_px), 270)                          # on the circle
    frame = np.zeros((540, 960, 3), np.uint8)
    cv2.circle(frame, real, 8, (0, 230, 240), -1)          # marker
    trk.find(frame, center, r_px)                          # lock onto marker
    cv2.circle(frame, (60, 500), 40, (0, 230, 240), -1)    # huge distractor
    tip, _ = trk.find(frame, center, r_px)
    stayed = tip is not None and abs(tip[0] - real[0]) < 5
    print(f"  [6] anti-jump  : stayed on marker={stayed} "
          f"(distractor at (60,500) ignored)")
    ok &= stayed

    # Test 7: FAST spin — marker moves ~150 px/frame; the prediction gate
    # must keep accepting it (this was the "stuck line" bug)
    trk = ColorTracker()
    eng = PhysicsEngine()
    w_fast, fps2 = 24.0, 30.0                              # 0.8 rad/frame
    hits = 0
    for i in range(90):
        t = i / fps2
        th = w_fast * t
        x = 480 + r_px * math.cos(th)
        y = 270 - r_px * math.sin(th)
        frame = np.zeros((540, 960, 3), np.uint8)
        cv2.circle(frame, (int(x), int(y)), 8, (0, 230, 240), -1)
        th_p = eng.predict_angle(t)
        predict = None
        if th_p is not None:
            predict = (480 + r_px * math.cos(th_p),
                       270 - r_px * math.sin(th_p))
        tip, _ = trk.find(frame, center, r_px, predict)
        if tip:
            hits += 1
            eng.add(t, tip[0], tip[1], 480, 270)
    w_meas = eng.omega_hist[-1]
    print(f"  [7] fast spin  : tracked {hits}/90 frames at 150 px/frame, "
          f"omega={w_meas:.3f} (true 24)")
    ok &= hits >= 88 and abs(w_meas - w_fast) < 0.3

    # Test 8: motion-blurred (desaturated) marker is caught via motion mask
    trk = ColorTracker()
    blurred = np.zeros((540, 960, 3), np.uint8)
    # dim, washed-out yellow streak (fails the strict range)
    cv2.ellipse(blurred, (680, 270), (34, 8), 0, 0, 360, (110, 150, 160), -1)
    motion = np.zeros((540, 960), np.uint8)
    cv2.circle(motion, (680, 270), 60, 255, -1)
    miss, _ = ColorTracker().find(blurred, center, r_px)   # strict only
    hit, _ = trk.find(blurred, center, r_px, None, motion)
    print(f"  [8] blur rescue: strict-only={miss}, with motion mask={hit}")
    ok &= miss is None and hit is not None and abs(hit[0] - 680) < 6

    print("PASS - physics pipeline accurate" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
def ask(prompt, default, cast=float):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print("  Invalid, using default.")
        return default


def main():
    ap = argparse.ArgumentParser(description="IB rotational mechanics tracker")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--video", type=str, default=None,
                    help="analyse a video file instead of the webcam")
    ap.add_argument("--length_cm", type=float, default=None,
                    help="staff length (cm)")
    ap.add_argument("--mass_g", type=float, default=None, help="staff mass (g)")
    ap.add_argument("--pivot", choices=["center", "end"], default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    print("=" * 56)
    print("  IB Physics HL - Rotational Mechanics Tracker")
    print("=" * 56)
    print("Enter your spinner's two measurements (press Enter for the")
    print("default). Everything else is done in the camera window.\n")
    if not HAVE_SCIPY:
        print("Note: SciPy not installed - using a simple finite-difference fallback.\n"
              "      Readouts will be noisier; `pip install scipy` for smoothed values.\n")
    L_cm = args.length_cm if args.length_cm else ask("Staff length (cm)", 8.0)
    m_g = args.mass_g if args.mass_g else ask("Staff mass  (g)", 20.0)
    pivot = args.pivot or "center"     # spinners rotate about their middle
    app = App(args.video if args.video else args.camera,
              L_cm / 100.0, m_g / 1000.0, pivot)
    print(f"\nUsing L = {L_cm:g} cm, m = {m_g:g} g, pivot = {pivot}.")
    print("Opening the camera window. Then:")
    print("  1) [M] click the marker once to lock its colour (recommended!)")
    print("  2) [C] click the centre of rotation, then the marker tip")
    print("  3) Spin!  [S] exports CSV, [V] records an annotated video.\n")
    sys.exit(app.run())


if __name__ == "__main__":
    main()
