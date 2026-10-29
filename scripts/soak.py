"""Explicit real-time replay soak; no camera, network, or audible output."""

import argparse
import asyncio
import json
import tempfile
import time
import tracemalloc
from pathlib import Path
from core.config import Settings
from core.runtime import Runtime


async def run(seconds):
    with tempfile.TemporaryDirectory() as directory:
        runtime = Runtime(Settings(data_dir=Path(directory), audio_enabled=False))
        await runtime.open()
        await runtime.start("replay")
        # Replace the finite UI replay producer with an indefinitely repeated fixture.
        runtime.session_task.cancel()
        await asyncio.gather(runtime.session_task, return_exceptions=True)
        runtime.session_task = None
        tracemalloc.start()
        start = time.monotonic()
        frame = 0
        try:
            while time.monotonic() - start < seconds:
                now = time.time()
                await runtime.ingest(
                    dict(
                        frame_id=frame,
                        captured_at=now,
                        face_detected=True,
                        ear=0.1 if frame % 300 < 60 else 0.3,
                        mar=0.2,
                        pitch=0,
                        yaw=0,
                        synthetic=True,
                    )
                )
                frame += 1
                await asyncio.sleep(0.033)
            current, peak = tracemalloc.get_traced_memory()
            print(
                json.dumps(
                    {
                        "kind": "real-time synthetic replay soak",
                        "seconds": time.monotonic() - start,
                        "frames": frame,
                        "python_current_bytes": current,
                        "python_peak_bytes": peak,
                        "health": runtime.health(),
                    },
                    indent=2,
                )
            )
        finally:
            await runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=3600)
    asyncio.run(run(parser.parse_args().seconds))
