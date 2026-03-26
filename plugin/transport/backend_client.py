from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple


class AsyncHTTPWorker:
    """Threaded HTTP worker with retry; no wx or pcbnew dependencies."""

    def __init__(self, request_timeout: int = 120, retries: int = 3, retry_base_s: float = 0.5):
        self.request_timeout = request_timeout
        self.retries = retries
        self.retry_base_s = retry_base_s
        self._q: queue.Queue = queue.Queue()
        self._results: Dict[str, Tuple[str, Any]] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            req_id, url, data = item
            result = self._do_request(url, data)
            with self._lock:
                self._results[req_id] = result

    def _do_request(self, url: str, data: Optional[Dict[str, Any]]) -> Tuple[str, Any]:
        last_err = "Unknown"
        for attempt in range(self.retries):
            if attempt > 0:
                time.sleep(self.retry_base_s * (2 ** (attempt - 1)))
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode() if data else None,
                    headers={"Content-Type": "application/json"},
                    method="POST" if data else "GET",
                )
                with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                    return "success", json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                    parsed = json.loads(body)
                    detail = parsed.get("detail", parsed)
                    last_err = f"HTTP {exc.code}: {detail}"
                except Exception:
                    last_err = f"HTTP {exc.code}: {exc.reason}"
                if 400 <= exc.code < 500:
                    break
            except Exception as exc:
                last_err = str(exc)
        return "error", last_err

    def request(self, url: str, data: Optional[Dict[str, Any]] = None) -> str:
        req_id = f"{time.monotonic():.9f}"
        self._q.put((req_id, url, data))
        return req_id

    def get_result(self, req_id: str) -> Tuple[str, Any]:
        with self._lock:
            return self._results.pop(req_id, ("pending", None))

    def close(self) -> None:
        self._q.put(None)


class BackendClient:
    """HTTP backend client with environment-driven base URL."""

    def __init__(self, base_url: Optional[str] = None, request_timeout: int = 120):
        self.base_url = (base_url or os.environ.get("AI_PCB_BACKEND_URL", "http://localhost:8765")).rstrip("/")
        self.worker = AsyncHTTPWorker(request_timeout=request_timeout)

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def request(self, path: str, data: Optional[Dict[str, Any]] = None) -> str:
        return self.worker.request(self.url(path), data)

    def get_result(self, req_id: str) -> Tuple[str, Any]:
        return self.worker.get_result(req_id)

    def health(self) -> bool:
        req_id = self.request("health")
        deadline = time.time() + 3
        while time.time() < deadline:
            status, result = self.get_result(req_id)
            if status == "pending":
                time.sleep(0.05)
                continue
            if status != "success":
                return False
            return result.get("status") in {"ok", "healthy", "degraded"}
        return False
