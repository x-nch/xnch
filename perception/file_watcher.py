import asyncio
import time

import httpx

from ..config import settings


class FileWatcher:
    def __init__(self, vault_dir: str | None = None, k8s_jobs_url: str | None = None) -> None:
        self._vault_dir = vault_dir or str(settings.vault_dir)
        self._k8s_jobs_url = k8s_jobs_url or "http://localhost:8001/k8s/jobs"
        self._last_event: dict[str, float] = {}
        self._debounce_s = 5.0
        self._observer = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()

        def _start():
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class VaultHandler(FileSystemEventHandler):
                def __init__(self, watcher: FileWatcher):
                    self._watcher = watcher

                def on_modified(self, event):
                    if event.is_directory:
                        return
                    asyncio.run_coroutine_threadsafe(
                        self._watcher._on_file_changed(event.src_path),
                        loop,
                    )

                def on_created(self, event):
                    if event.is_directory:
                        return
                    asyncio.run_coroutine_threadsafe(
                        self._watcher._on_file_changed(event.src_path),
                        loop,
                    )

            self._observer = Observer()
            self._observer.schedule(
                VaultHandler(self),
                self._vault_dir,
                recursive=True,
            )
            self._observer.start()

        await loop.run_in_executor(None, _start)

    async def stop(self) -> None:
        if self._observer is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._observer.stop)
            self._observer = None

    async def _on_file_changed(self, path: str) -> None:
        now = time.time()
        last = self._last_event.get(path, 0.0)
        if now - last < self._debounce_s:
            return
        self._last_event[path] = now
        payload = {
            "file_path": path,
            "event": "modified",
            "timestamp": now,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                await client.post(self._k8s_jobs_url, json=payload)
            except Exception:
                pass
