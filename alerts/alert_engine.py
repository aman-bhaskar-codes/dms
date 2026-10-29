"""Local serialized macOS audio, with critical preemption and failure status."""

import asyncio
import shutil
import sys


class AlertEngine:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.queue = asyncio.PriorityQueue(maxsize=8)
        self.process = None
        self.task = None
        self.sequence = 0
        self.last_error = ""

    def start(self):
        self.task = asyncio.create_task(self._run())

    def speak(self, text, critical=False):
        if not self.enabled:
            return
        if critical:
            if self.process and self.process.returncode is None:
                self.process.terminate()
            # Obsolete coaching and announcements must not follow a critical alert.
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
        self.sequence += 1
        if not self.queue.full():
            self.queue.put_nowait((0 if critical else 10, self.sequence, text[:1000]))

    def silence(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()

    async def _run(self):
        while True:
            _, _, text = await self.queue.get()
            try:
                if sys.platform != "darwin" or not shutil.which("say"):
                    self.last_error = "macOS speech is unavailable; visual alerts remain active"
                    continue
                self.process = await asyncio.create_subprocess_exec(
                    "say",
                    text,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                exit_code = await self.process.wait()
                if exit_code not in (0, -15):
                    self.last_error = "Audio output failed; visual alerts remain active"
                elif exit_code == 0:
                    self.last_error = ""
            except Exception:
                self.last_error = "Audio output failed; visual alerts remain active"
            finally:
                self.process = None
                self.queue.task_done()

    async def shutdown(self):
        self.silence()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
