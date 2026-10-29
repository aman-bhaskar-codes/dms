"""Offline signal-policy load test; explicitly not a webcam or inference benchmark."""

import argparse
import json
import time
import tracemalloc
from core.monitoring import Monitor
from agents.safety_agent import SafetyPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=108000)
args = parser.parse_args()
monitor = Monitor()
policy = SafetyPolicy()
latencies = []
alerts = 0
tracemalloc.start()
start = time.perf_counter()
for index in range(args.frames):
    t = index / 30
    tick = time.perf_counter()
    snapshot = monitor.update(
        dict(
            frame_id=index,
            captured_at=t,
            face_detected=True,
            ear=0.1 if index % 900 < 60 else 0.3,
            mar=0.2,
            pitch=0,
            yaw=0,
            synthetic=True,
        )
    )
    event = policy.evaluate(snapshot, now=t)
    alerts += event is not None and event.transition == "triggered"
    if index >= 100:
        latencies.append((time.perf_counter() - tick) * 1000)
    if len(latencies) > 3600:
        latencies.pop(0)
_, peak = tracemalloc.get_traced_memory()
elapsed = time.perf_counter() - start
print(
    json.dumps(
        {
            "kind": "offline signal-policy load test; no camera inference",
            "frames": args.frames,
            "simulated_seconds": args.frames / 30,
            "wall_seconds": elapsed,
            "throughput_frames_s": args.frames / elapsed,
            "p95_policy_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
            "peak_python_bytes": peak,
            "retained_window_samples": len(monitor.window),
            "alerts": alerts,
        },
        indent=2,
    )
)
