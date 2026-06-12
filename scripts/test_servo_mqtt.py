"""Publish pan commands directly to MQTT — tests ESP8266 without face tracking."""
from __future__ import annotations

import argparse
import time

import paho.mqtt.client as mqtt

DEFAULT_BROKER = "157.173.101.159"
DEFAULT_TOPIC = "vision/faithlock/movement"


def main() -> int:
    parser = argparse.ArgumentParser(description="MQTT servo smoke test")
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument("--direction", choices=["right", "left", "center"], default="right")
    args = parser.parse_args()

    cmd = {
        "right": "MOVED_RIGHT",
        "left": "MOVED_LEFT",
        "center": "CENTERED",
    }[args.direction]

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="servo-smoke-test")
    client.connect(args.broker, 1883, 60)
    print(f"Publishing {cmd} x{args.steps} to {args.topic} @ {args.broker}")
    for i in range(args.steps):
        client.publish(args.topic, cmd, qos=0)
        print(f"  step {i + 1}/{args.steps}")
        time.sleep(args.interval)
    client.publish(args.topic, "STOPPED", qos=0)
    client.disconnect()
    print("Done. Servo should have moved visibly if ESP is on Wi-Fi and flashed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
