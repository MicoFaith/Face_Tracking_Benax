"""
BENAX single-speaker camera tracking protocol.

Motor commands (MQTT payloads):
  MOVED_LEFT, MOVED_RIGHT, CENTERED, STOPPED, SCAN, OUT_OF_FRAME

Legacy short names (LEFT, RIGHT, CENTER, IDLE, SEARCH) are still accepted by firmware.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Canonical BENAX motor commands
MOVED_LEFT = "MOVED_LEFT"
MOVED_RIGHT = "MOVED_RIGHT"
CENTERED = "CENTERED"
STOPPED = "STOPPED"
SCAN = "SCAN"
OUT_OF_FRAME = "OUT_OF_FRAME"

ALL_MOTOR_COMMANDS = (
    MOVED_LEFT,
    MOVED_RIGHT,
    CENTERED,
    STOPPED,
    SCAN,
    OUT_OF_FRAME,
)

# Map legacy MQTT names to BENAX names (for backward compatibility)
LEGACY_TO_BENAX: Dict[str, str] = {
    "LEFT": MOVED_LEFT,
    "RIGHT": MOVED_RIGHT,
    "CENTER": CENTERED,
    "IDLE": STOPPED,
    "SEARCH": SCAN,
    "STOP": STOPPED,
    "SCAN": SCAN,
}

BENAX_TO_LEGACY: Dict[str, str] = {
    MOVED_LEFT: "LEFT",
    MOVED_RIGHT: "RIGHT",
    CENTERED: "CENTER",
    STOPPED: "IDLE",
    SCAN: "SEARCH",
    OUT_OF_FRAME: "SEARCH",
}


# Commands the ESP8266 servo firmware actually acts on (pulse pan / hold / scan).
SERVO_ACTION_COMMANDS = (MOVED_LEFT, MOVED_RIGHT, CENTERED, STOPPED, SCAN)


def motor_command_for_servo(message: str) -> Optional[str]:
    """
    Map internal tracking state to MQTT payload for the servo.

    OUT_OF_FRAME and other non-action states are not sent — the servo keeps
    its current angle (avoids jitter from occlusion messages).
    """
    cmd = normalize_motor_command(message)
    if cmd == OUT_OF_FRAME:
        return None
    if cmd in SERVO_ACTION_COMMANDS:
        return cmd
    return STOPPED


def normalize_motor_command(message: str) -> str:
    """Normalize incoming or outgoing motor command to BENAX canonical form."""
    clean = str(message or "").strip().upper()
    if clean in ALL_MOTOR_COMMANDS:
        return clean
    return LEGACY_TO_BENAX.get(clean, STOPPED)


def command_from_error_with_hysteresis(
    error_x: float,
    deadzone_px: float,
    center_exit_hysteresis_px: float,
    previous_command: str,
) -> str:
    """Convert horizontal tracking error into a BENAX motor command."""
    prev = normalize_motor_command(previous_command)
    dz = float(deadzone_px)
    hyst = float(center_exit_hysteresis_px)
    err = float(error_x)

    if prev == CENTERED:
        if abs(err) <= dz + hyst:
            return CENTERED
    elif prev == MOVED_LEFT:
        if err > dz + hyst:
            return MOVED_RIGHT
        if abs(err) <= dz:
            return CENTERED
        return MOVED_LEFT
    elif prev == MOVED_RIGHT:
        if err < -(dz + hyst):
            return MOVED_LEFT
        if abs(err) <= dz:
            return CENTERED
        return MOVED_RIGHT

    if abs(err) <= dz:
        return CENTERED
    if err < 0:
        return MOVED_LEFT
    return MOVED_RIGHT


def command_when_speaker_lost(
    lost_for_sec: float,
    search_delay_sec: float,
    *,
    hold_command: str = CENTERED,
    allow_scan: bool = False,
) -> str:
    """Occlusion handling: hold last pan command by default (no SCAN interrupt)."""
    hold = normalize_motor_command(hold_command)
    if not allow_scan:
        if hold in (MOVED_LEFT, MOVED_RIGHT):
            return hold
        return CENTERED
    if lost_for_sec < float(search_delay_sec):
        return OUT_OF_FRAME
    return SCAN


def tracking_phase_label(
    *,
    speaker_name: Optional[str],
    locked: bool,
    speaker_visible: bool,
    motor_command: str,
) -> str:
    """Human-readable pipeline stage for dashboard."""
    if not speaker_name:
        return "enroll"
    if not locked:
        return "lock"
    if speaker_visible:
        cmd = normalize_motor_command(motor_command)
        if cmd in (MOVED_LEFT, MOVED_RIGHT):
            return "following"
        return "tracking"
    cmd = normalize_motor_command(motor_command)
    if cmd == SCAN:
        return "searching"
    return "reacquire"


@dataclass
class EvidenceLogger:
    """CSV + JSONL evidence log for assessment validation."""

    logs_dir: Path = field(default_factory=lambda: Path("logs/evidence"))
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    _csv_path: Optional[Path] = field(default=None, init=False)
    _jsonl_path: Optional[Path] = field(default=None, init=False)
    _csv_file: Optional[object] = field(default=None, init=False)
    _csv_writer: Optional[csv.DictWriter] = field(default=None, init=False)
    _fieldnames: List[str] = field(default_factory=lambda: [
        "timestamp_iso",
        "unix_ts",
        "speaker_id",
        "confidence",
        "similarity",
        "distance",
        "motor_command",
        "error_x_px",
        "locked",
        "speaker_visible",
        "faces_in_frame",
        "ignored_faces",
        "fps",
        "event_note",
    ])

    def __post_init__(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.logs_dir / f"session_{self.session_id}.csv"
        self._jsonl_path = self.logs_dir / f"session_{self.session_id}.jsonl"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
        self._csv_writer.writeheader()
        self._csv_file.flush()
        print(f"[Evidence] Logging to {self._csv_path}")

    def log(
        self,
        *,
        speaker_id: Optional[str],
        confidence: float,
        similarity: float,
        distance: float,
        motor_command: str,
        error_x_px: float,
        locked: bool,
        speaker_visible: bool,
        faces_in_frame: int,
        ignored_faces: int = 0,
        fps: Optional[float] = None,
        event_note: str = "",
    ) -> None:
        now = time.time()
        row = {
            "timestamp_iso": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            "unix_ts": round(now, 3),
            "speaker_id": speaker_id or "",
            "confidence": round(float(confidence), 4),
            "similarity": round(float(similarity), 4),
            "distance": round(float(distance), 4),
            "motor_command": normalize_motor_command(motor_command),
            "error_x_px": round(float(error_x_px), 2),
            "locked": bool(locked),
            "speaker_visible": bool(speaker_visible),
            "faces_in_frame": int(faces_in_frame),
            "ignored_faces": int(ignored_faces),
            "fps": round(float(fps), 2) if fps is not None else "",
            "event_note": str(event_note or ""),
        }
        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
            if self._csv_file is not None:
                self._csv_file.flush()
        if self._jsonl_path is not None:
            with open(self._jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(row) + "\n")

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
