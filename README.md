# Rotational Mechanics Lab — IB Physics HL

Real-time webcam tracker for a spinning staff. Overlays θ, ω, α, v = ωr,
a_c = ω²r, τ = Iα and rolling ω–t / τ–t graphs directly on the live feed,
with proper physics vector arrows (tangential **v**, inward **a꜀**, curved
spin-direction arrow, θ angle arc from the reference axis).

## Run it

Double-click **run_tracker.bat**, or:

```
python spin_tracker.py
```

It asks two things only: staff **length in cm** (default 8) and **mass in
grams** (default 20). Pivot is assumed at the staff's centre (tip radius =
L/2, I = mL²/12). If yours pivots at one end, add `--pivot end`.

## In the window (3 steps)

1. Put bright tape on ONE tip of the staff. Press **M**, then click the tape
   once — this locks the exact colour (strongly recommended: it stops the
   tracker confusing the toy's body for the marker).
2. Press **C**, click the **centre of rotation** (the axle), then click the
   tape tip. This calibrates pixels → centimetres.
3. Spin. Everything updates live.

## Keys

| Key | Action |
|---|---|
| C | Calibrate (click axle, then tip) |
| M | Lock marker colour from a click |
| SPACE | Pause / resume |
| R | Reset angle, graphs, data |
| S | Save all recorded data to CSV (for your IA graphs) |
| V | Record an annotated MP4 of the tracked feed |
| Q / ESC | Quit |

## How the tracking stays locked at speed

Three layers keep the marker from being lost or stolen:

1. **Annulus constraint** — after calibration, the tip can physically only be
   on the spin circle, so the search is restricted to a thin ring around it.
   Background objects and the toy's body are simply never considered.
2. **Physics prediction** — the tracker extrapolates where the tip *should*
   be from the current ω and looks there first, so a marker that moves 150 px
   between frames is still followed instead of rejected as a "jump".
3. **Motion-blur rescue** — fast spins smear the tape and wash out its
   colour. A relaxed colour range is allowed, but only inside the
   frame-difference motion mask, so it can't latch onto anything static.

The badge in the top-right shows the live tracking state:
**TRACKING LOCKED** (green) / **SEARCHING…** / **CALIBRATE**.

## If the spin is too fast for your webcam

At 30 fps the marker must move less than half a turn per frame
(max |ω| ≈ 94 rad s⁻¹ ≈ 15 rev/s). Beyond that, record slow-motion video on
a phone (120/240 fps) and analyse the file — timing then comes from the
video's frame rate, so all values stay physically correct:

```
python spin_tracker.py --video clip.mp4 --length_cm 8 --mass_g 20
```

Calibrate on the first frame (M, C), press SPACE to analyse. A CSV is saved
automatically at the end; press **V** first if you also want the annotated
video written out.

## Verify the math anytime

```
python spin_tracker.py --selftest
```

Eight checks feed synthetic spins of known ω and α through the identical
pipeline: constant spin, constant α, clockwise sign, pixel noise, colour
detection, background-distractor rejection, 150 px/frame fast tracking, and
motion-blur rescue.

## Tips for clean data

- Bright, even lighting — short exposures freeze motion blur.
- Pick a tape colour nothing else in frame shares (M-click to lock it).
- Keep the camera square-on to the plane of rotation; a tilted view turns
  the circle into an ellipse and distorts ω within each revolution.
- ω and α are Savitzky–Golay smoothed, so readouts lag ~2 frames.
