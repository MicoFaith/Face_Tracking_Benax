"""Check MQTT broker reachability from this PC (same path ESP8266 uses)."""
from __future__ import annotations

import argparse
import socket
import sys
import time

import paho.mqtt.client as mqtt

DEFAULT_BROKER = "157.173.101.159"
DEFAULT_TOPIC = "vision/faithlock/movement"


def tcp_check(host: str, port: int, timeout: float = 5.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError as exc:
        print(f"TCP {host}:{port} FAILED — {exc}")
        return False
    finally:
        sock.close()


def mqtt_check(host: str, port: int, topic: str) -> bool:
    connected = {"ok": False, "rc": -1}

    def on_connect(client, _userdata, _flags, reason_code, _properties=None):
        connected["ok"] = reason_code == 0
        connected["rc"] = reason_code

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="faithlock-diagnose")
    client.on_connect = on_connect
    try:
        client.connect(host, port, keepalive=45)
    except Exception as exc:
        print(f"MQTT connect() exception: {exc}")
        return False

    client.loop_start()
    deadline = time.time() + 10.0
    while time.time() < deadline and not connected["ok"]:
        time.sleep(0.2)

    if not connected["ok"]:
        print(f"MQTT connect failed (rc={connected['rc']})")
        client.loop_stop()
        client.disconnect()
        return False

    print(f"MQTT publish test -> {topic}")
    client.publish(topic, "CENTERED", qos=0)
    time.sleep(0.5)
    client.loop_stop()
    client.disconnect()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose FaithLock MQTT broker")
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    args = parser.parse_args()

    print(f"Broker: {args.broker}:{args.port}")
    print(f"Topic:  {args.topic}\n")

    if not tcp_check(args.broker, args.port):
        print("\nESP8266 rc=-2 (CONNECT_FAILED) usually means this TCP step fails from Wi-Fi.")
        print("Check: broker online, firewall allows port 1883, ESP on same 2.4GHz network.")
        return 1

    print(f"TCP {args.broker}:{args.port} OK")

    if not mqtt_check(args.broker, args.port, args.topic):
        return 1

    print("\nMQTT OK from this PC.")
    print("If ESP still shows rc=-2: re-flash firmware, confirm Year3D Wi-Fi, wait for [WiFi] Connected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
