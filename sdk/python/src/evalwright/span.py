"""Span: one timed unit of work (an LLM call, a tool call, a task, a scorer, an eval case)."""
from __future__ import annotations

import os
import random
import threading
import time
import traceback
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .tracer import Tracer

SpanType = Literal["task", "llm", "tool", "function", "eval", "score", "http", "retrieval", "custom"]

_rng = random.Random(int.from_bytes(os.urandom(16), "big"))
_rng_lock = threading.Lock()


def _reseed() -> None:   # after fork, children must not repeat the parent's ids
    _rng.seed(int.from_bytes(os.urandom(16), "big"))


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reseed)


def new_trace_id() -> str:
    with _rng_lock:
        return f"{_rng.getrandbits(128):032x}"


def new_span_id() -> str:
    with _rng_lock:
        return f"{_rng.getrandbits(64):016x}"


class Span:
    """Mutable while open; serialized once at end(). Cheap when not sampled."""

    __slots__ = (
        "_c0",
        "_ended",
        "_lock",
        "_t0",
        "_tracer",
        "cpu_ns",
        "duration_ns",
        "end_ns",
        "error",
        "expected",
        "input",
        "metadata",
        "metrics",
        "name",
        "output",
        "parent_id",
        "sampled",
        "scores",
        "span_id",
        "start_ns",
        "status",
        "tags",
        "trace_id",
        "type",
    )
    trace_id: str
    span_id: str
    parent_id: str | None
    sampled: bool

    def __init__(self, tracer: Tracer | None, name: str, type: SpanType = "function",
                 parent: Span | None = None, sampled: bool = True, trace_id: str | None = None) -> None:
        self._tracer = tracer
        self.name, self.type = name, type
        self.trace_id = parent.trace_id if parent is not None else (trace_id or new_trace_id())
        self.parent_id = parent.span_id if parent is not None else None
        self.span_id = new_span_id()
        self.sampled = parent.sampled if parent is not None else sampled
        self.start_ns = time.time_ns()
        self._t0 = time.perf_counter_ns()
        self._c0 = time.thread_time_ns()
        self.end_ns: int | None = None
        self.duration_ns: int | None = None
        self.cpu_ns: int | None = None
        self.input: Any = None
        self.output: Any = None
        self.expected: Any = None
        self.metadata: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.scores: dict[str, float | None] = {}
        self.tags: list[str] = []
        self.status: Literal["ok", "error"] = "ok"
        self.error: dict[str, str] | None = None
        self._ended = False
        self._lock = threading.Lock()

    # ---- user API -------------------------------------------------------------------
    def log(self, *, input: Any = None, output: Any = None, expected: Any = None,
            metadata: Mapping[str, Any] | None = None, metrics: Mapping[str, float] | None = None,
            scores: Mapping[str, float | None] | None = None, tags: list[str] | None = None) -> Span:
        """Attach data. Later calls merge into earlier ones; None means "leave unchanged"."""
        if not self.sampled:
            return self
        with self._lock:
            if input is not None:
                self.input = input
            if output is not None:
                self.output = output
            if expected is not None:
                self.expected = expected
            if metadata:
                self.metadata.update(metadata)
            if metrics:
                self.metrics.update({k: float(v) for k, v in metrics.items()})
            if scores:
                self.scores.update(scores)
            if tags:
                self.tags.extend(t for t in tags if t not in self.tags)
        return self

    def set_error(self, exc: BaseException) -> None:
        self.status = "error"
        if self.sampled:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8))
            self.error = {"type": type(exc).__name__, "message": str(exc)[:2000], "stack": tb[-4000:]}

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        self.end_ns = time.time_ns()
        self.duration_ns = time.perf_counter_ns() - self._t0
        self.cpu_ns = time.thread_time_ns() - self._c0
        if self._tracer is not None and self.sampled:
            self._tracer._on_end(self)

    @property
    def ended(self) -> bool:
        return self._ended

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "span_id": self.span_id, "parent_id": self.parent_id,
                "name": self.name, "type": self.type, "start_ns": self.start_ns, "end_ns": self.end_ns,
                "duration_ms": None if self.duration_ns is None else self.duration_ns / 1e6,
                "cpu_ms": None if self.cpu_ns is None else self.cpu_ns / 1e6,
                "input": self.input, "output": self.output, "expected": self.expected,
                "metadata": self.metadata, "metrics": self.metrics, "scores": self.scores, "tags": self.tags,
                "status": self.status, "error": self.error}

    def __repr__(self) -> str:
        return f"Span({self.name!r}, type={self.type}, trace={self.trace_id[:8]}, status={self.status})"


class _NoopSpan(Span):
    """Shared stand-in for disabled tracers and sampled-out traces: every method is a no-op,
    children of it are no-ops too, and creating one costs nothing."""

    def __init__(self) -> None:
        self.trace_id = "0" * 32
        self.span_id = "0" * 16
        self.parent_id = None
        self.name, self.type, self.sampled = "noop", "custom", False
        self.status, self.error = "ok", None
        self.metadata, self.metrics, self.scores, self.tags = {}, {}, {}, []
        self.input = self.output = self.expected = None
        self.start_ns = self._t0 = self._c0 = 0
        self.end_ns = self.duration_ns = self.cpu_ns = None
        self._tracer, self._ended = None, False
        self._lock = threading.Lock()

    def log(self, **_: Any) -> Span:
        return self

    def set_error(self, exc: BaseException) -> None:
        pass

    def end(self) -> None:
        pass


NOOP_SPAN: Span = _NoopSpan()
