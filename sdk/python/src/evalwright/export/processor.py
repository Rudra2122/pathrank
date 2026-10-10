"""Background batching processor: the reason tracing can't hurt the host application.

* on_end() is O(1) and lock-light: append to a bounded deque, maybe notify the worker.
  When the queue is full the OLDEST span is dropped (and counted), never the caller blocked.
* A daemon worker exports batches of up to max_batch every flush_interval or when a batch fills.
* Retryable failures (network, 408/429/5xx) back off exponentially with full jitter, honoring
  Retry-After; non-retryable failures drop the batch (counted).
* A circuit breaker opens after N consecutive failed batches, pauses exports for a cooldown,
  then lets one probe batch through (half-open).
* fork-safe: a forked child gets a fresh lock/condition/empty queue and starts its own worker.
* flush(timeout) waits for everything queued before the call to be exported or given up on.
"""
from __future__ import annotations

import collections
import contextlib
import logging
import os
import random
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any

from .base import Exporter, ExportResult

logger = logging.getLogger("evalwright")
_instances: weakref.WeakSet[BatchProcessor] = weakref.WeakSet()


@dataclass
class RetryPolicy:
    max_retries: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0

    def delay(self, attempt: int, retry_after: float | None, rng: random.Random) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.max_delay)
        return rng.uniform(0.0, min(self.max_delay, self.base_delay * (2 ** attempt)))   # full jitter


@dataclass
class CircuitBreaker:
    threshold: int = 5
    cooldown: float = 30.0
    failures: int = 0
    opened_at: float | None = None
    clock: Any = field(default=time.monotonic, repr=False)

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.cooldown else "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record(self, ok: bool) -> None:
        if ok:
            self.failures, self.opened_at = 0, None
        else:
            self.failures += 1
            if self.failures >= self.threshold or self.state == "half_open":
                self.opened_at = self.clock()


class BatchProcessor:
    def __init__(self, exporter: Exporter, *, max_queue: int = 10_000, max_batch: int = 512,
                 flush_interval: float = 1.0, retry: RetryPolicy | None = None,
                 breaker: CircuitBreaker | None = None, seed: int | None = None) -> None:
        if max_batch <= 0 or max_queue <= 0:
            raise ValueError("max_batch and max_queue must be positive")
        self.exporter = exporter
        self.max_queue, self.max_batch, self.flush_interval = max_queue, max_batch, flush_interval
        self.retry = retry or RetryPolicy()
        self.breaker = breaker or CircuitBreaker()
        self._rng = random.Random(seed)
        self._counters = {"enqueued": 0, "exported": 0, "dropped_queue_full": 0, "dropped_failed": 0,
                          "retries": 0, "export_calls": 0, "batches_failed": 0}
        self._last_error: str | None = None
        self._init_runtime()
        _instances.add(self)

    def _init_runtime(self) -> None:
        self._q: collections.deque[dict[str, Any]] = collections.deque()
        self._cv = threading.Condition(threading.Lock())
        self._thread: threading.Thread | None = None
        self._stop = False
        self._stop_event = threading.Event()     # interrupts backoff sleeps (only shutdown sets it)
        self._shutdown_deadline: float | None = None
        self._inflight = 0
        self._seq_in = 0          # spans accepted
        self._seq_done = 0        # spans exported or dropped
        self._pid = os.getpid()

    # ---- producer side -------------------------------------------------------------------
    def on_end(self, record: dict[str, Any]) -> None:
        if os.getpid() != self._pid:      # forked without register_at_fork (defensive)
            self._init_runtime()
        with self._cv:
            if self._stop:
                return
            if len(self._q) >= self.max_queue:
                self._q.popleft()
                self._counters["dropped_queue_full"] += 1
                self._seq_done += 1
            self._q.append(record)
            self._seq_in += 1
            self._counters["enqueued"] += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="evalwright-exporter", daemon=True)
                self._thread.start()
            if len(self._q) >= self.max_batch:
                self._cv.notify()

    # ---- worker ------------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cv:
                if not self._q and not self._stop:
                    self._cv.wait(self.flush_interval)
                if self._stop and not self._q:
                    return
                if not self._q:
                    continue
                if not self.breaker.allow():
                    self._cv.wait(min(1.0, self.breaker.cooldown))
                    continue
                n = min(self.max_batch, len(self._q))
                batch = [self._q.popleft() for _ in range(n)]
                self._inflight = n
            ok = self._export_with_retry(batch)
            with self._cv:
                self._inflight = 0
                self._seq_done += len(batch)
                if ok:
                    self._counters["exported"] += len(batch)
                else:
                    self._counters["dropped_failed"] += len(batch)
                    self._counters["batches_failed"] += 1
                self._cv.notify_all()

    def _export_with_retry(self, batch: list[dict[str, Any]]) -> bool:
        attempt = 0
        while True:
            self._counters["export_calls"] += 1
            try:
                res = self.exporter.export(batch)
            except Exception as e:  # noqa: BLE001 - exporter bugs must not kill the worker
                res = ExportResult.retry(f"{type(e).__name__}: {e}")
            if res.success:
                self.breaker.record(True)
                return True
            self._last_error = res.error
            if not res.retryable or attempt >= self.retry.max_retries:
                self.breaker.record(False)
                logger.debug("evalwright: dropping batch of %d: %s", len(batch), res.error)
                return False
            delay = self.retry.delay(attempt, res.retry_after, self._rng)
            attempt += 1
            self._counters["retries"] += 1
            # sleep on a dedicated event: flush() notifications must not shorten backoff
            if self._stop_event.wait(delay) and self._deadline_passed():
                self.breaker.record(False)
                return False

    def _deadline_passed(self) -> bool:
        d = self._shutdown_deadline
        return d is not None and time.monotonic() >= d

    # ---- control -------------------------------------------------------------------------------
    def flush(self, timeout: float | None = 5.0) -> bool:
        """Block until everything enqueued before this call is exported or dropped."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            target = self._seq_in
            self._cv.notify_all()
            while self._seq_done < target:
                if self._thread is None or not self._thread.is_alive():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cv.notify_all()
                self._cv.wait(0.05 if remaining is None else min(0.05, remaining))
        return True

    def shutdown(self, timeout: float | None = 5.0) -> None:
        self._shutdown_deadline = None if timeout is None else time.monotonic() + timeout
        self.flush(timeout)
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=0.5 if timeout is None else max(0.0, min(timeout, 1.0)))
        with contextlib.suppress(Exception):
            self.exporter.close()

    def stats(self) -> dict[str, Any]:
        with self._cv:
            return {**self._counters, "queued": len(self._q), "inflight": self._inflight,
                    "circuit": self.breaker.state, "last_error": self._last_error}


def _after_fork_in_child() -> None:
    for p in list(_instances):
        p._init_runtime()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
