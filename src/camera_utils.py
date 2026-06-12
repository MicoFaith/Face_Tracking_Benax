"""Camera frame orientation and low-latency capture helpers."""
from __future__ import annotations

import cv2
import numpy as np

ROTATE_MODES = ("none", "180", "90cw", "90ccw", "flip-v", "flip-h")


def configure_low_latency_capture(
    cap: cv2.VideoCapture,
    width: int = 640,
    height: int = 480,
) -> tuple[int, int]:
    """Request a small buffer and moderate resolution for live pan tracking."""
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return actual_w, actual_h


def read_latest_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    """Drop stale buffered frames and return the newest image."""
    if not cap.grab():
        return False, None
    # Drain one extra frame when the driver queues more than one image.
    if cap.grab():
        return cap.retrieve()
    return cap.retrieve()


def apply_camera_orientation(frame: np.ndarray, mode: str) -> np.ndarray:
    """Rotate or flip a BGR frame before detection / display."""
    key = str(mode or "none").strip().lower()
    if key in ("", "none"):
        return frame
    if key == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    if key in ("90cw", "cw"):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if key in ("90ccw", "ccw"):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if key == "flip-v":
        return cv2.flip(frame, 0)
    if key == "flip-h":
        return cv2.flip(frame, 1)
    raise ValueError(f"Unknown camera orientation {mode!r}. Use one of: {', '.join(ROTATE_MODES)}")
