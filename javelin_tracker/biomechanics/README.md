# Biomechanics Module

This package contains the motion-analysis stack for javelin throws: pose detection, video utilities, configuration, and (future) metrics/feedback.

## Submodules
- `config.py`: Centralized thresholds, storage paths, phase timings, outlier limits; supports env + TOML/JSON overrides; `BIOMECHANICS_LOGGER`.
- `pose_estimation/pose_detector.py`: MediaPipe Holistic wrapper returning 33 pose landmarks, hand data, confidences; thread-safe; context-managed.
- `utils/video.py`: OpenCV helpers for validation, metadata, and frame extraction (generator-friendly).
- `metrics/`, `comparison/`, `database/`, `feedback/`: Stubs for metrics computation, session comparisons, persistence, and coaching cues.

## Architecture (high level)
```
                +--------------------+
                |  config & logging  |
                +---------+----------+
                          |
          +---------------v---------------+
          |      utils.video (IO)         |
          +---------------+---------------+
                          |
                  frames (BGR)
                          |
          +---------------v---------------+
          | pose_estimation.PoseDetector  |
          +---------------+---------------+
                          |
                landmarks + confidences
                          |
          +-------v--------+--------v------+
          |    metrics     |   feedback    |
          +-------+--------+--------+------+
                  |                 |
             reports/db        cues/UX
```

## Quick Start (dev)
1) Install deps: `pip install -r requirements.txt`.
2) Validate a sample video: `python -c "from javelin_tracker.biomechanics.utils import validate_video_readable; print(validate_video_readable('demo/sample.avi'))"`.
3) Run pose detection on a frame:
```python
import cv2
from javelin_tracker.biomechanics.pose_estimation import PoseDetector

frame = cv2.imread("demo/frame.jpg")
with PoseDetector() as detector:
    result = detector.process_frame(frame)
print(result["pose_confidence_avg"], len(result["landmarks"]))
```
4) Run tests: `pytest tests/test_biomechanics.py`.

## Improving limb tracking (pose quality)
The pose stack uses MediaPipe:
- If `mediapipe.solutions` is available, it uses **Holistic** with internal landmark smoothing.
- Otherwise it uses the **Tasks PoseLandmarker** in **VIDEO** mode for better temporal tracking.

You can switch the PoseLandmarker model variant (accuracy vs speed):
```bash
# Higher accuracy (slower)
export THROWS_TRACKER_POSE_LANDMARKER_MODEL_VARIANT=heavy   # or full

# Fastest
export THROWS_TRACKER_POSE_LANDMARKER_MODEL_VARIANT=lite
```

By default, the app uses `heavy` (accuracy-first) for sports clips. Use `lite` if you need speed.

For fully offline installs, point to a local `.task` file:
```bash
export THROWS_TRACKER_POSE_LANDMARKER_MODEL_PATH=/absolute/path/to/pose_landmarker_full.task
```

You can force the MediaPipe backend:
```bash
# Default (recommended): Tasks PoseLandmarker
export THROWS_TRACKER_POSE_BACKEND=pose_landmarker

# Legacy backend (includes hand landmarks, but less robust in some clips)
export THROWS_TRACKER_POSE_BACKEND=holistic
```

### Occlusion handling (arm behind head)
Consumer videos often have fast motion + self-occlusion (arm behind head) which can cause pose models to "snap" a wrist/elbow to the wrong pixel.

This project includes two optional stabilizers:
- **ROI tracking / zoom-in** (default ON): crops around the athlete each frame to increase effective resolution.
- **Kinovea-style optical flow refinement** (default ON): uses OpenCV optical flow to keep landmarks continuous when the pose estimate jumps.
- **AI athlete segmentation**: uses PoseLandmarker segmentation masks to down-weight landmarks that fall outside the athlete silhouette (helps prevent tracking the javelin/background). Note: on Python 3.12 + MediaPipe Tasks, segmentation is disabled by default to avoid a known hard-crash in some builds; you can force-enable via `THROWS_TRACKER_POSE_SEGMENTATION=true`.

Tuning knobs:
```bash
# Disable ROI cropping if it causes misses (default: true)
export THROWS_TRACKER_POSE_ROI=false

# ROI selection mode (default: "torso")
# - "torso": use stable torso/upper-limb points to avoid ROI drift onto the javelin/background
# - "all": use all landmarks (can be less stable if distal landmarks snap incorrectly)
export THROWS_TRACKER_POSE_ROI_MODE=torso

# Minimum ROI crop size in pixels (default: 160)
export THROWS_TRACKER_POSE_ROI_MIN_SIZE=384

# ROI smoothing (0..1). Higher = more stable ROI (default: 0.45)
export THROWS_TRACKER_POSE_ROI_SMOOTHING=0.8

# Disable optical-flow refinement (default: true)
export THROWS_TRACKER_POSE_OPTICAL_FLOW=false

# Optical-flow tuning: make it more/less aggressive when the pose model reports high confidence
# but the landmark jumps (common when an occluded wrist "snaps" to a wrong pixel).
export THROWS_TRACKER_POSE_OPTICAL_FLOW_HIGH_CONF_MULTIPLIER=2.25

# Disable segmentation-based athlete isolation (default: true)
export THROWS_TRACKER_POSE_SEGMENTATION=false

# Segmentation mask tuning (Tasks backend)
export THROWS_TRACKER_POSE_SEGMENTATION_THRESHOLD=0.4
export THROWS_TRACKER_POSE_SEGMENTATION_DILATE_PX=8
export THROWS_TRACKER_POSE_SEGMENTATION_MIN_PIXELS=250

# Segmentation cleanup (helps when the mask includes the javelin / spurious regions)
# These only kick in when the mask bbox looks very sparse or extremely elongated.
export THROWS_TRACKER_POSE_SEGMENTATION_OPEN_PX=2
export THROWS_TRACKER_POSE_MASK_SPUR_RATIO_THRESHOLD=3.5
export THROWS_TRACKER_POSE_MASK_SPUR_ASPECT_THRESHOLD=3.8
export THROWS_TRACKER_POSE_MASK_COMPONENT_KEEP_RATIO=0.06
export THROWS_TRACKER_POSE_MASK_MIN_AREA_FRACTION_AFTER_REFINE=0.55

# Segmentation bbox trimming (helps when the mask includes long thin spurs like the javelin)
# Only applies when the mask bbox looks like it has an extreme tail.
export THROWS_TRACKER_POSE_MASK_BBOX_TRIM_QUANTILE=0.01
export THROWS_TRACKER_POSE_MASK_BBOX_IQR_FACTOR=4.0

# If multiple people are in-frame, increase pose candidates (default: 2)
export THROWS_TRACKER_POSE_NUM_POSES=3

# Confidence thresholds for Tasks PoseLandmarker (0..1)
export THROWS_TRACKER_POSE_MIN_DETECTION_CONFIDENCE=0.4
export THROWS_TRACKER_POSE_MIN_PRESENCE_CONFIDENCE=0.4
export THROWS_TRACKER_POSE_MIN_TRACKING_CONFIDENCE=0.4

# For Holistic backend only (0..2). Higher = more accurate (slower).
export THROWS_TRACKER_POSE_HOLISTIC_MODEL_COMPLEXITY=2
```

### "Slow motion" tracking (timestamp scaling)
MediaPipe Tasks uses timestamps for temporal tracking. For very fast throws, you can make the tracker assume *smaller* motion between frames by shrinking the timestamp deltas (this does **not** change the underlying video frames).

```bash
# Option A: pretend the clip is higher-FPS (smaller dt between frames)
export THROWS_TRACKER_POSE_ASSUMED_FPS=120

# Option B: scale timestamps directly (<1.0 = smaller dt, often more stable)
export THROWS_TRACKER_POSE_TIME_SCALE=0.5
```

The Tasks PoseLandmarker backend defaults to `THROWS_TRACKER_POSE_TIME_SCALE=0.5` for stability. Set it to `1.0` to disable, or try `0.25` for even more aggressive slow-motion tracking on very fast clips.

Note: the web UI playback speed control only affects viewing; it does not change analysis results.

## Camera angle variability (real-world videos)
In the real world, throws are filmed from different directions (side/front/quarter views) and with different zoom levels.

This project reduces camera-dependence by:
- Computing most joint angles from **3D landmark vectors** (x/y/z) rather than image-plane angles.
- Converting normalized landmark coordinates to **pixel space** (when video width/height are available) so kinematics are less distorted by aspect ratio.
- Normalizing positions/velocities by **athlete body height** (body-heights/second, or meters/second when athlete height is known).
- Using **trunk-axis twist** (hip/shoulder separation) instead of 2D line-angle differences for rotation-style metrics.

Limitations:
- Monocular 3D (single camera) is still an estimate; extreme camera angles and heavy occlusions can reduce accuracy.
- For best results, film with a stable camera and keep the athlete large in frame (side or quarter-side views work well).

## Adding New Metrics
- Place computation helpers in `metrics/` (e.g., joint angle time series, release velocity).
- Consume pose data from `PoseDetector` (landmarks and confidences).
- Use thresholds from `config.py` (e.g., `CONFIDENCE_THRESHOLD`, `OUTLIER_THRESHOLDS`) and log via `BIOMECHANICS_LOGGER`.
- Expose public functions via `metrics/__init__.py` and add unit tests under `tests/`.

## References (concepts)
- Javelin throw phases: approach → delivery → release (phase timings configurable in `config.py`).
- Key joints tracked: shoulders, elbows, wrists, hips, knees (MediaPipe Holistic 33-point model).
- Confidence handling: per-landmark visibility/presence; frame validity gates via config thresholds.
