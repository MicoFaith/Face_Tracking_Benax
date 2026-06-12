#!/usr/bin/env python3
"""
BENAX integrated system validation.

Runs automated checks and writes a traceable validation report under logs/evidence/.
Run from repo root:

    python addons/mqtt_servo_tracking/validate_system.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.speaker_protocol import (
    CENTERED,
    MOVED_LEFT,
    MOVED_RIGHT,
    OUT_OF_FRAME,
    SCAN,
    STOPPED,
    command_from_error_with_hysteresis,
    command_when_speaker_lost,
    motor_command_for_servo,
    normalize_motor_command,
)
from src.speaker_recognition import FaceDisposition, classify_face


def check(name: str, ok: bool, detail: str = "") -> dict:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))
    return {"check": name, "status": status, "detail": detail}


def main() -> int:
    print("=" * 60)
    print("BENAX Integrated System Validation")
    print("=" * 60)

    results: list[dict] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = REPO_ROOT / "logs" / "evidence"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"validation_report_{ts}.json"

    # --- Models ---
    onnx = REPO_ROOT / "models" / "embedder_arcface.onnx"
    land = REPO_ROOT / "models" / "face_landmarker.task"
    results.append(check("Model: embedder_arcface.onnx", onnx.exists(), str(onnx)))
    results.append(check("Model: face_landmarker.task", land.exists(), str(land)))

    # --- Database ---
    db = REPO_ROOT / "data" / "db" / "face_db.npz"
    results.append(check("Speaker database exists", db.exists(), "Run: python -m src.enroll"))

    # --- Single-speaker recognition policy ---
    sp = classify_face("Alice", "Alice", True, 0.82, 0.18)
    results.append(check(
        "Speaker-only: authorized match",
        sp.disposition == FaceDisposition.AUTHORIZED_SPEAKER and sp.label == "Alice",
        f"label={sp.label}",
    ))
    other = classify_face("Alice", "Bob", True, 0.75, 0.25)
    results.append(check(
        "Speaker-only: other face ignored",
        other.disposition == FaceDisposition.IGNORED_OTHER and other.label == "Ignored",
        f"label={other.label}",
    ))
    unk = classify_face("Alice", None, False, 0.2, 0.8)
    results.append(check(
        "Speaker-only: unknown face",
        unk.disposition == FaceDisposition.UNKNOWN,
        f"label={unk.label}",
    ))

    # --- Track -> command pipeline ---
    left = command_from_error_with_hysteresis(-120, 80, 30, CENTERED)
    results.append(check("Tracking: negative error -> MOVED_LEFT", left == MOVED_LEFT, left))
    right = command_from_error_with_hysteresis(120, 80, 30, CENTERED)
    results.append(check("Tracking: positive error -> MOVED_RIGHT", right == MOVED_RIGHT, right))
    cen = command_from_error_with_hysteresis(5, 80, 30, CENTERED)
    results.append(check("Tracking: deadband -> CENTERED", cen == CENTERED, cen))
    oof = command_when_speaker_lost(0.3, 0.8, allow_scan=True)
    results.append(check("Occlusion: brief loss -> OUT_OF_FRAME", oof == OUT_OF_FRAME, oof))
    scan = command_when_speaker_lost(1.5, 0.8, allow_scan=True)
    results.append(check("Re-acquire: extended loss -> SCAN", scan == SCAN, scan))
    hold = command_when_speaker_lost(2.0, 0.8, hold_command=MOVED_LEFT, allow_scan=False)
    results.append(check("Smooth follow: hold pan when lost", hold == MOVED_LEFT, hold))
    results.append(check("Servo MQTT: OUT_OF_FRAME not sent", motor_command_for_servo(OUT_OF_FRAME) is None, "filtered"))
    results.append(check(
        "Motor command normalization",
        normalize_motor_command("LEFT") == MOVED_LEFT,
        "legacy LEFT supported",
    ))

    # --- Hardware / software modules present ---
    modules = [
        ("MQTT tracker", REPO_ROOT / "addons/mqtt_servo_tracking/recognize_mqtt.py"),
        ("ESP8266 firmware", REPO_ROOT / "addons/mqtt_servo_tracking/esp8266/face_tracker_servo/face_tracker_servo.ino"),
        ("Dashboard", REPO_ROOT / "dashboard/index.html"),
        ("Speaker protocol", REPO_ROOT / "src/speaker_protocol.py"),
        ("Speaker recognition", REPO_ROOT / "src/speaker_recognition.py"),
        ("Enrollment", REPO_ROOT / "src/enroll.py"),
    ]
    for label, path in modules:
        results.append(check(f"Module: {label}", path.exists(), str(path.relative_to(REPO_ROOT))))

    # --- Dependencies ---
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import paho.mqtt.client  # noqa: F401
        results.append(check("Python dependencies import", True))
    except ImportError as e:
        results.append(check("Python dependencies import", False, str(e)))

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    report = {
        "generated_at": datetime.now().isoformat(),
        "system": "BENAX AI-Powered Single-Speaker Face Recognition and Camera Tracking",
        "summary": {"passed": passed, "failed": failed, "total": len(results)},
        "realistic_test_scenarios": [
            {
                "scenario": "Multiple faces in frame",
                "expected": "Only enrolled speaker labeled; others show Ignored; motor tracks speaker only",
                "evidence_field": "ignored_faces > 0 in logs/evidence/session_*.csv",
            },
            {
                "scenario": "Speaker temporarily occluded",
                "expected": "OUT_OF_FRAME then SCAN; re-lock same speaker when visible",
                "evidence_field": "event_note contains occlusion / reacquire_scan",
            },
            {
                "scenario": "Speaker moves left/right",
                "expected": "MOVED_LEFT / MOVED_RIGHT / CENTERED with error_x_px logged",
                "evidence_field": "motor_command + error_x_px columns",
            },
            {
                "scenario": "No speaker lock active",
                "expected": "STOPPED published; servo idle",
                "evidence_field": "motor_command=STOPPED, locked=false",
            },
        ],
        "safety_controls": [
            "Servo angle constrained SERVO_MIN_ANGLE..SERVO_MAX_ANGLE in firmware",
            "ESP COMMAND_TIMEOUT_MS stops motion if MQTT stream drops",
            "PC deadband + hysteresis reduces jitter",
            "command_confirm_frames debounces LEFT/RIGHT flicker",
            "Dedicated 5V servo supply recommended (shared GND with ESP)",
            "STOPPED on unlock/session end",
        ],
        "checks": results,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Report: {report_path}")
    print(f"Summary: {passed}/{len(results)} passed, {failed} failed")
    print()
    print("Manual demonstration checklist:")
    for i, sc in enumerate(report["realistic_test_scenarios"], 1):
        print(f"  {i}. {sc['scenario']}")
        print(f"     Expected: {sc['expected']}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
