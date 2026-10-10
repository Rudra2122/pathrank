"""Opt-in function-level profiling attached to spans.

    @ew.traced(profile=True)              # profile every call of this function
    def rank(user): ...

    with ew.span("batch"), ew.profiled(top=10):   # or profile any block into the current span
        run_batch()

The span gets metadata["profile"]: the top functions by cumulative time, e.g.
    [{"function": "pair_features (features.py:61)", "calls": 1, "cumtime_ms": 41.2, "tottime_ms": 3.1}, ...]

Rules that keep it safe to leave in production code:
  * only sampled spans are profiled (sampled-out and disabled spans cost nothing extra)
  * one profiler per thread: nested profiled spans are covered by the outermost one
  * if another profiler is already active (pytest-profiling, an IDE, sys.monitoring tools), skip
    silently instead of raising
  * evalwright's own frames are left out of the report
  * on asyncio, cProfile sees everything that runs on the thread while the span is open, including
    other tasks; treat async profiles as "what the event loop did during this span"
"""
from __future__ import annotations

import contextlib
import cProfile
import os
import pstats
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .span import Span

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_local = threading.local()


def _report(prof: cProfile.Profile, top: int) -> list[dict[str, Any]]:
    st = pstats.Stats(prof)
    rows = []
    for (filename, line, func), (_cc, nc, tt, ct, _callers) in st.stats.items():  # type: ignore[attr-defined]
        if filename.startswith(_PKG_DIR) or func in ("<module>", "disable") or "cProfile" in func:
            continue
        where = f"{os.path.basename(filename)}:{line}" if filename != "~" else "builtin"
        rows.append({"function": f"{func} ({where})", "calls": int(nc),
                     "cumtime_ms": round(ct * 1000, 3), "tottime_ms": round(tt * 1000, 3)})
    rows.sort(key=lambda r: r["cumtime_ms"], reverse=True)
    return rows[:top]


@contextmanager
def profile_into(span: Span, top: int = 15) -> Iterator[None]:
    if not span.sampled or getattr(_local, "active", False):
        yield
        return
    prof = cProfile.Profile()
    try:
        prof.enable()
    except ValueError:            # another profiling tool owns this thread
        yield
        return
    _local.active = True
    try:
        yield
    finally:
        prof.disable()
        _local.active = False
        with contextlib.suppress(Exception):     # profiling must never break the traced code
            span.log(metadata={"profile": _report(prof, top)})


@contextmanager
def profiled(top: int = 15) -> Iterator[None]:
    """Profile the enclosed block into the current span (no-op outside a span)."""
    from .tracer import current_span

    span = current_span()
    if span is None:
        yield
        return
    with profile_into(span, top):
        yield
