"""Video utilities for frame extraction and metadata inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, Union

import cv2
import numpy as np

from javelin_tracker.biomechanics import BIOMECHANICS_LOGGER

logger = BIOMECHANICS_LOGGER


def validate_video_readable(video_path: Union[str, Path]) -> Dict[str, Union[bool, int, float, str]]:
    """Check if a video can be opened and read; raise descriptive errors otherwise.

    Example:
        >>> validate_video_readable("throws.mp4")  # doctest: +SKIP
        {'readable': True, 'width': 1920, 'height': 1080, 'fps': 60.0, 'total_frames': 900, 'codec': 'avc1'}
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a video file, but got a directory: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video {path}. The file may be corrupted or use an unsupported codec."
        )

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = _decode_fourcc(codec_int)

        if not codec.strip("\x00"):
            raise ValueError(
                f"Unable to determine codec for {path}. The file may use an unsupported or missing codec."
            )

        if fps <= 0:
            raise ValueError(
                f"Invalid FPS reported for {path}. The file may be corrupted or unreadable (fps={fps})."
            )

        if width <= 0 or height <= 0 or total_frames <= 0:
            raise ValueError(
                f"Invalid metadata for {path}. The file may be corrupted or unreadable (w={width}, h={height}, frames={total_frames})."
            )

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            raise ValueError(
                f"Failed to read the first frame from {path}. The file may be corrupted or use an unsupported codec."
            )

        return {
            "readable": True,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "codec": codec,
        }
    finally:
        cap.release()


def extract_frames(
    video_path: Union[str, Path],
    fps: float = 30.0,
    skip_frames: int = 0,
    output_format: str = "BGR",
    resize_to: Optional[Tuple[int, int]] = None,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yield frames from a video as (frame_idx, timestamp_ms, frame).

    Designed for memory efficiency on large videos.
    Supports optional color format conversion (BGR, RGB, GRAY) and resizing.

    Example:
        >>> from pathlib import Path
        >>> for idx, ts_ms, frame in extract_frames(Path("throws.mp4"), fps=30, skip_frames=2, output_format="RGB"):
        ...     print(idx, ts_ms, frame.shape)
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a video file, but got a directory: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video {path}. The file may be corrupted or use an unsupported codec."
        )

    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    effective_fps = meta_fps if meta_fps and meta_fps > 0 else fps

    if effective_fps <= 0:
        cap.release()
        raise ValueError(f"Unable to determine FPS for {path}; verify the file is readable.")

    logger.info(
        "Starting frame extraction for %s at ~%s fps with skip_frames=%s",
        path,
        effective_fps,
        skip_frames,
    )

    frame_idx = 0
    last_timestamp_ms: float | None = None
    try:
        ret, frame = cap.read()
        if not ret:
            raise ValueError(
                f"Could not read the first frame from {path}. The file may be corrupted or unsupported."
            )

        while ret:
            # Prefer container timestamps when available (helps variable-FPS clips);
            # fall back to an fps-derived timestamp if the backend returns 0/NaN.
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if not np.isfinite(pos_msec):
                pos_msec = 0.0
            timestamp_ms = pos_msec
            if last_timestamp_ms is not None and timestamp_ms + 0.5 < last_timestamp_ms:
                # Some codecs/backends report non-monotonic timestamps; ignore them.
                timestamp_ms = 0.0
            if timestamp_ms <= 0.0:
                timestamp_ms = (frame_idx / effective_fps) * 1000.0
            last_timestamp_ms = float(timestamp_ms)

            processed_frame = _convert_frame_color(frame, output_format)
            processed_frame = _resize_frame(processed_frame, resize_to)
            yield frame_idx, timestamp_ms, processed_frame

            if frame_idx % 200 == 0:
                logger.debug("Processed frame %s from %s", frame_idx, path.name)

            frame_idx += 1

            skipped = 0
            while skipped < skip_frames:
                if not cap.grab():
                    logger.warning("Stopped during skip at frame %s for %s", frame_idx, path.name)
                    return
                frame_idx += 1
                skipped += 1

            ret, frame = cap.read()

        logger.info("Completed frame extraction for %s; total frames processed: %s", path, frame_idx)
    finally:
        cap.release()


def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
    """Return basic metadata for the provided video.

    Example:
        >>> meta = get_video_metadata("throws.mp4")
        >>> meta["fps"], meta["duration_seconds"]
    """
    validation = validate_video_readable(video_path)
    path = Path(video_path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not reopen video {path} after validation. The file may be corrupted or use an unsupported codec."
        )

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = (total_frames / fps) if fps and fps > 0 else 0.0

        return {
            "width": validation["width"],
            "height": validation["height"],
            "fps": validation["fps"] if validation["fps"] and validation["fps"] > 0 else fps,
            "total_frames": validation["total_frames"],
            "duration_seconds": duration_seconds,
            "codec": validation["codec"],
        }
    finally:
        cap.release()


def _convert_frame_color(frame: np.ndarray, output_format: str) -> np.ndarray:
    fmt = (output_format or "BGR").upper()
    if fmt == "BGR":
        return frame
    if fmt == "RGB":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if fmt in ("GRAY", "GREY"):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported output_format '{output_format}'. Use BGR, RGB, or GRAY.")


def _resize_frame(frame: np.ndarray, resize_to: Optional[Tuple[int, int]]) -> np.ndarray:
    if not resize_to:
        return frame
    if (
        not isinstance(resize_to, tuple)
        or len(resize_to) != 2
        or any((not isinstance(dim, int) or dim <= 0) for dim in resize_to)
    ):
        raise ValueError("resize_to must be a (width, height) tuple of positive ints.")
    width, height = resize_to
    return cv2.resize(frame, (width, height))


def _decode_fourcc(cc: int) -> str:
    return "".join([chr((cc >> 8 * i) & 0xFF) for i in range(4)])
