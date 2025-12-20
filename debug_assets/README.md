## Biomechanics Debug Assets (Repro Protocol)

This folder is for **deterministic reproduction** of biomechanics tracking bugs
(e.g., “anchor points don’t attach to joints”, wrist/elbow snapping to the
background/javelin, occlusion failures when the arm goes behind the head).

Important:
- Do **not** commit copyrighted videos or extracted overlay frames.
- The repo `.gitignore` intentionally ignores `debug_assets/videos/*`,
  `debug_assets/overlays_expected/*`, and `debug_assets/runs/`.

### Folder layout

```
debug_assets/
  README.md
  videos/                 # local-only clips (ignored by git)
  metadata/               # small JSON metadata files (safe to commit)
  overlays_expected/      # local-only overlay frame exports (ignored by git)
  runs/                   # local-only pipeline outputs (ignored by git)
```

### Clip naming convention (required)

Store clips under `debug_assets/videos/` using:

`debug_{good|hard}_v{N}_{YYYYMMDD}_{HEIGHT}p_{FPS}fps_{view}_{camera}.mp4`

Examples:
- `debug_good_v1_20251220_1080p_60fps_side_static.mp4`
- `debug_hard_v1_20251220_720p_30fps_front_handheld.mp4`

Rules:
- Never overwrite an existing `*_vN_*` file; bump the version instead.
- Keep the clip short (6–10s) and centered on **delivery → release**.

### Specs

#### “Good” debug clip (deterministic baseline)
Use this to reproduce the bug in the cleanest conditions (minimize video-caused noise):
- Duration: **6–10s** (last 2–3 strides + delivery + release + ~0.5s follow-through)
- Resolution: **≥720p** (1080p preferred)
- FPS: **≥30** (60 preferred; avoid variable-FPS if you can)
- Camera: **static/tripod**, no zoom
- View: **side** or **3/4 side**, full body in frame
- People: **exactly 1** visible athlete (no passers-by behind them)
- Lighting: bright/even, no flicker/backlight silhouette
- Clothing: high contrast vs background; avoid loose sleeves
- Background: low texture (avoid fences/trees directly behind the throwing arm)

Why these matter:
- Pose models need pixels on wrists/elbows; low resolution + blur makes joints ambiguous.
- High fps reduces per-frame displacement, improving temporal tracking.
- Camera motion confounds optical flow and ROI tracking.
- Multi-person and clutter cause candidate switching and background “latching”.

#### “Hard” regression clip (stress test)
Use later to ensure fixes generalize:
- Mild handheld sway or panning allowed
- Front-ish view where throwing arm passes behind head/torso for ≥10 frames
- Cluttered background (crowd/trees/fence), mixed contrast clothing
- Lower fps (30) and/or variable fps acceptable

### Checklist: “good enough” for debugging

Video properties:
- [ ] 6–10s and includes delivery→release clearly
- [ ] ≥720p and not heavily compressed
- [ ] ≥30 fps and timestamps are stable/monotonic
- [ ] Camera is static (no zoom)

Visual clarity:
- [ ] Wrist/elbow/shoulder are distinguishable around release
- [ ] No severe motion blur at release
- [ ] Athlete occupies a meaningful portion of the frame

Scene constraints:
- [ ] Only one person is in the frame
- [ ] Background behind the throwing arm is not highly textured

Determinism:
- [ ] Two consecutive runs show the issue at roughly the same timestamp (±2 frames)

### How to freeze a clip for reproducible debugging

1) Trim the source to the throw window (6–10s, delivery→release).
2) Export as H.264 MP4 with constant fps (avoid VFR).
3) Register it (writes metadata + enforces naming):

```bash
.venv/bin/python scripts/biomech_repro.py register \
  --clip-type good \
  --source /path/to/your_trimmed.mp4 \
  --view side \
  --camera static
```

4) Validate it (sanity checks):

```bash
.venv/bin/python scripts/biomech_repro.py validate debug_assets/videos/debug_good_v1_*.mp4
```

5) Run the pipeline and export overlay frames for visual proof:

```bash
THROWS_TRACKER_POSE_TIME_SCALE=0.25 \
.venv/bin/python scripts/biomech_repro.py run debug_assets/videos/debug_good_v1_*.mp4
```

Outputs are written under `debug_assets/runs/<clip-stem>/...` (ignored by git).

### Acceptance criteria (repro)

- Bug reproduces within **30 seconds** of running the pipeline on the “good” clip.
- Bug is visible in exported overlay frames (wrist/elbow/shoulder anchor drifts onto background/javelin or wrong limb for ≥5 consecutive frames).
