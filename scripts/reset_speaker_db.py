"""Delete all enrolled speakers and crops for a fresh single-speaker setup."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "db"
ENROLL_DIR = ROOT / "data" / "enroll"


def main() -> int:
    removed: list[str] = []

    for path in (DB_DIR / "face_db.npz", DB_DIR / "face_db.json"):
        if path.exists():
            path.unlink()
            removed.append(str(path.relative_to(ROOT)))

    if ENROLL_DIR.exists():
        shutil.rmtree(ENROLL_DIR)
        removed.append(str(ENROLL_DIR.relative_to(ROOT)))
    ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    if removed:
        print("Cleared:")
        for item in removed:
            print(f"  - {item}")
    else:
        print("Nothing to clear — database was already empty.")

    print("\nNext — enroll Faith (save photos when ready):")
    print("  python -m src.enroll --name Faith --camera-index 1 --camera-rotate 180 --fresh")
    print("\nThen start tracking:")
    print("  python addons\\mqtt_servo_tracking\\recognize_mqtt.py --cpu-only --camera-index 1 --speaker-name Faith")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
