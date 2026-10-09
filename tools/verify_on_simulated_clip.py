"""
End-to-end accuracy check on a SIMULATED clip with known kinematics.

`spin_tracker.py --selftest` feeds synthetic points straight into the physics
engine. This goes one step further: it renders a video of a yellow-tipped staff
spinning with a known angular velocity, runs the app's REAL main loop on it
(HSV tracking, calibration, physics, CSV export) with the window and mouse
replaced by a script, and compares the exported CSV against the ground truth.

    python tools/verify_on_simulated_clip.py                  # prints PASS / FAIL
    python tools/verify_on_simulated_clip.py --shot out.png   # also save the overlay at 6.7 s

The clip: omega ramps 0 -> 14 rad/s over 3 s (alpha = 14/3 rad/s^2), then stays
at 14 rad/s. Staff 38 cm, 20 g, pivot at the centre (r = 19 cm), 30 fps.
Nothing is written into the repository; all files go to a temporary folder.
"""

import argparse
import csv
import glob
import importlib.util
import math
import os
import sys
import tempfile

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W, H, FPS, SECS = 960, 540, 30, 12
CX, CY, R_PX = 480, 270, 190
RAMP_S, OMEGA_END = 3.0, 14.0
ALPHA = OMEGA_END / RAMP_S
LENGTH_M, MASS_KG = 0.38, 0.020           # r = L/2 = 0.19 m, I = m L^2 / 12


def true_theta(t):
    return 0.5 * ALPHA * t * t if t < RAMP_S else 0.5 * ALPHA * RAMP_S ** 2 + OMEGA_END * (t - RAMP_S)


def render_clip(path):
    bg = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        bg[y] = (70 + y // 14, 62 + y // 14, 58 + y // 12)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(SECS * FPS + 1):
        th = true_theta(i / FPS)
        tip = (int(CX + R_PX * math.cos(th)), int(CY - R_PX * math.sin(th)))
        tail = (int(CX - R_PX * math.cos(th)), int(CY + R_PX * math.sin(th)))
        f = bg.copy()
        cv2.line(f, tail, tip, (60, 90, 140), 14, cv2.LINE_AA)       # brown staff
        cv2.circle(f, (CX, CY), 18, (90, 90, 90), -1, cv2.LINE_AA)   # hub
        cv2.circle(f, tip, 15, (0, 232, 242), -1, cv2.LINE_AA)       # yellow tape (BGR)
        out.write(f)
    out.release()


def run_app_headless(clip, workdir, shot_path, shot_frame):
    spec = importlib.util.spec_from_file_location("spin_tracker", os.path.join(REPO, "spin_tracker.py"))
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)

    tape = (CX + R_PX, CY)                      # where the tape is on frame 0
    steps = [("key", ord("m")), ("click", *tape),                    # lock the marker colour
             ("key", ord("c")), ("click", CX, CY), ("click", *tape)]  # axle, then tip
    app = st.App(clip, LENGTH_M, MASS_KG, "center")
    box = {"cb": None, "n": 0}

    cv2.namedWindow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.setMouseCallback = lambda win, cb: box.__setitem__("cb", cb)

    def imshow(win, canvas):
        box["n"] += 1
        if shot_path and box["n"] == shot_frame:
            cv2.imwrite(shot_path, canvas)
    cv2.imshow = imshow

    def wait_key(_ms=1):
        if steps:
            kind, *rest = steps.pop(0)
            if kind == "click":
                box["cb"](cv2.EVENT_LBUTTONDOWN, rest[0], rest[1], 0, None)
                return 255
            return rest[0]
        return ord("q") if app.msg.startswith("End of video") else 255
    cv2.waitKey = wait_key

    cwd = os.getcwd()
    os.chdir(workdir)                           # the app saves its CSV into the cwd
    try:
        app.run()
    finally:
        os.chdir(cwd)
    return sorted(glob.glob(os.path.join(workdir, "spin_data_*.csv")))[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", help="save the app's overlay (frame --shot-frame) to this PNG")
    ap.add_argument("--shot-frame", type=int, default=200)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "simulated_spin.mp4")
        render_clip(clip)
        csv_path = run_app_headless(clip, tmp, os.path.abspath(args.shot) if args.shot else None, args.shot_frame)
        rows = list(csv.reader(open(csv_path)))
    t, theta, omega, alpha, v, a_c, tau = np.array(rows[1:], float).T

    r_m, inertia = LENGTH_M / 2, MASS_KG * LENGTH_M ** 2 / 12
    steady, ramp = t > 4.0, (t > 0.5) & (t < 2.5)
    total_angle = theta[-1] - theta[0]
    true_angle = true_theta(t[-1]) - true_theta(t[0])
    checks = [
        # name, measured, true, relative tolerance
        ("omega, steady phase (rad/s)", np.median(omega[steady]), OMEGA_END, 0.01),
        ("v = omega r (m/s)", np.median(v[steady]), OMEGA_END * r_m, 0.01),
        ("a_c = omega^2 r (m/s^2)", np.median(a_c[steady]), OMEGA_END ** 2 * r_m, 0.02),
        ("alpha, ramp phase (rad/s^2)", np.median(alpha[ramp]), ALPHA, 0.05),
        ("tau = I alpha, ramp (N m)", np.median(tau[ramp]), inertia * ALPHA, 0.05),
        ("total angle (rad)", total_angle, true_angle, 0.005),
    ]
    ok = True
    print(f"{'quantity':32s} {'measured':>11s} {'true':>11s} {'error':>8s}")
    for name, got, want, tol in checks:
        err = abs(got - want) / abs(want)
        good = err <= tol
        ok &= good
        print(f"{name:32s} {got:11.4g} {want:11.4g} {err:7.2%}  {'ok' if good else 'FAIL'}")
    print("PASS - the real app loop recovers the known kinematics" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
