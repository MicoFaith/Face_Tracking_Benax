"""Probe OpenCV camera indices and save a snapshot per device."""
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "camera_probe"
OUT.mkdir(parents=True, exist_ok=True)


def probe(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "default"

    for _ in range(8):
        cap.read()
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return {"index": index, "read": False, "width": w, "height": h, "backend": backend}

    path = OUT / f"camera_index_{index}.jpg"
    cv2.imwrite(str(path), frame)
    return {
        "index": index,
        "read": True,
        "width": w,
        "height": h,
        "shape": frame.shape,
        "backend": backend,
        "snapshot": str(path),
    }


def main():
    max_index = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Scanning camera indices 0..{max_index} (Windows DirectShow)\n")

    found = []
    for i in range(max_index + 1):
        r = probe(i)
        if r is None:
            print(f"  Index {i}: not available")
            continue
        found.append(r)
        if r["read"]:
            print(
                f"  Index {i}: OK  {r['width']}x{r['height']}  "
                f"frame={r['shape'][1]}x{r['shape'][0]}  backend={r['backend']}"
            )
            print(f"           snapshot -> {r['snapshot']}")
        else:
            print(f"  Index {i}: opened but could not read a frame")

    print()
    if not found:
        print("No cameras detected.")
        return 1

    readable = [r for r in found if r["read"]]
    print(f"Found {len(readable)} usable camera(s).")
    if len(readable) == 1:
        idx = readable[0]["index"]
        print(f"\nUse for tracking: --camera-index {idx}")
    elif len(readable) >= 2:
        print("\nMultiple cameras found. Open the snapshots in logs/camera_probe/:")
        for r in readable:
            print(f"  index {r['index']}: {r['snapshot']}")
        print(
            "\nFor pan-servo tracking, use the index whose snapshot is the "
            "EXTERNAL camera mounted on the servo (not the laptop webcam)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
