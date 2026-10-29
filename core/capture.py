"""Latest-only sequential processing with a bounded result buffer."""

import asyncio
from concurrent.futures import ThreadPoolExecutor


class LatestProcessor:
    def __init__(self, process, cleanup=None):
        self.process, self.cleanup = process, cleanup
        self.queue = asyncio.Queue(maxsize=1)
        self.results = asyncio.Queue(maxsize=1)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentinel-perception")
        self.dropped = 0
        self.closed = False
        self.done = asyncio.Event()

    def submit(self, frame):
        if self.closed:
            return
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait(frame)

    async def run(self):
        loop = asyncio.get_running_loop()
        try:
            while True:
                frame = await self.queue.get()
                if frame is None:
                    break
                result = await loop.run_in_executor(self.executor, self.process, frame)
                if self.results.full():
                    self.results.get_nowait()
                self.results.put_nowait(result)
        finally:
            if self.cleanup:
                await loop.run_in_executor(self.executor, self.cleanup)
            self.executor.shutdown(wait=True)
            self.done.set()

    async def close(self):
        if not self.closed:
            self.closed = True
            if self.queue.full():
                self.queue.get_nowait()
            self.queue.put_nowait(None)
        await self.done.wait()
