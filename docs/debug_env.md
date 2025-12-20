# Freeze Runtime Environment (Biomechanics Debugging)

This project processes videos using OpenCV + MediaPipe (Tasks PoseLandmarker). Pose tracking bugs can be **very sensitive** to:
- OS + CPU/GPU backend
- Python version
- MediaPipe/OpenCV wheel variants
- transitive dependency updates (numpy/scipy/etc.)

This doc defines a *single reproducibility protocol* so the same debug clip reproduces the same bug across machines and time.

---

## Deliverable 1: Version capture checklist

Capture these whenever you report a tracking bug:

### System
- OS name + version (macOS/Windows/Linux)
- Kernel version (or build number)
- CPU model + core count
- GPU model + driver/runtime (if applicable)

### Python
- Python version (`python -V`)
- Pip version (`python -m pip -V`)
- Virtualenv type (venv/conda) and path (optional)

### Packages (must be exact)
- `mediapipe`
- `opencv-python`
- `numpy`
- `scipy`
- `pandas`
- `flask` (webapp runtime)

### App settings that affect tracking
- `THROWS_TRACKER_POSE_LANDMARKER_MODEL_VARIANT`
- `THROWS_TRACKER_POSE_TIME_SCALE`
- `THROWS_TRACKER_POSE_ASSUMED_FPS`
- `THROWS_TRACKER_POSE_ROI`, `THROWS_TRACKER_POSE_OPTICAL_FLOW`

Use the capture script below to write a single JSON snapshot.

---

## Deliverable 2: Recommended pinning strategy (golden format)

### Golden format: `requirements.lock` (pip-freeze snapshot)

For a solo developer, the fastest dependable approach is to:
- keep `requirements.txt` as the *human-maintained* dependency list
- commit a **fully pinned** `requirements.lock` that represents the known-good environment that reproduces the bug

Why this works well solo:
- zero extra tooling (no pip-tools/conda required)
- pins **transitive** dependencies (critical for MediaPipe/OpenCV stability)
- easy acceptance test: `pip install -r requirements.lock` then verify versions match

Platform notes:
- MediaPipe/OpenCV wheels are platform-specific. A lock file generated on macOS may not resolve on Windows/Linux.
- If you need cross-platform reproduction, generate **one lock per platform** using the same format, e.g.:
  - `requirements.lock.darwin-py312.txt`
  - `requirements.lock.linux-py312.txt`
  - `requirements.lock.windows-py312.txt`

This repo ships a default `requirements.lock` intended for the primary dev platform.

---

## Deliverable 3: Verification procedure

### A) Capture environment snapshot

```bash
.venv/bin/python scripts/debug_env.py capture --out debug_assets/metadata/debug_env_snapshot.json
```

### B) Freeze (generate/update the lock)

```bash
.venv/bin/python scripts/debug_env.py freeze --out requirements.lock
```

### C) Verify the current environment matches the lock

```bash
.venv/bin/python scripts/debug_env.py verify --lock requirements.lock
```

### D) End-to-end sanity run (debug clip)

```bash
THROWS_TRACKER_POSE_TIME_SCALE=0.25 \
.venv/bin/python scripts/biomech_repro.py run debug_assets/videos/debug_good_v1_*.mp4
```

This must complete without import errors and export overlay frames under `debug_assets/runs/...`.

---

## Deliverable 4: Acceptance tests

### Fresh install reproducibility

**Acceptance:** two fresh installs produce identical package versions.

Procedure:
1) Create two new venvs (same Python version), install lock, install project:
   ```bash
   python -m venv .venv_a
   . .venv_a/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.lock
   pip install -e .
   python -m pip freeze > /tmp/freeze_a.txt
   deactivate

   python -m venv .venv_b
   . .venv_b/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.lock
   pip install -e .
   python -m pip freeze > /tmp/freeze_b.txt
   deactivate

   diff -u /tmp/freeze_a.txt /tmp/freeze_b.txt
   ```
2) `diff` output should be empty (identical).

### Debug clip runs end-to-end

**Acceptance:** the debug clip runs end-to-end without dependency errors and exports overlay frames.

```bash
.venv/bin/python scripts/debug_env.py verify --lock requirements.lock
THROWS_TRACKER_POSE_TIME_SCALE=0.25 .venv/bin/python scripts/biomech_repro.py run debug_assets/videos/debug_good_v1_*.mp4
```

---

## Platform-specific pitfalls (OpenCV/MediaPipe)

### macOS
- Prefer Python 3.12 + `mediapipe==0.10.x` Tasks wheels.
- Avoid mixing Homebrew OpenCV with `opencv-python` in the same runtime.
- If you see hard crashes related to segmentation masks, keep segmentation off on Py3.12 Tasks builds (this repo defaults it off for stability).

### Windows
- Ensure you’re using 64-bit Python.
- If MediaPipe wheels fail, pin to a known compatible Python minor version and regenerate `requirements.lock`.

### Linux
- If you use NVIDIA, keep driver/toolkit consistent; prefer CPU-only for deterministic debugging unless GPU is required.
- Avoid mixing distro OpenCV + pip OpenCV.

---

## Minimal policy: upgrades and regression confirmation

1) Only upgrade deps deliberately:
- Create a branch.
- Update `requirements.txt` (direct deps only).
- Re-freeze: `scripts/debug_env.py freeze --out requirements.lock`.

2) Confirm no regressions:
- Run tests: `.venv/bin/python -m pytest -q`
- Run the “good” debug clip repro and export overlays.
- Then run the “hard” clip for stress/regression.

3) If a bug changes behavior:
- Treat it as a breaking change to the repro environment and bump the clip version or record the new lock.

