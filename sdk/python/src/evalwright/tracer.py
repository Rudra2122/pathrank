"""Tracer: creates spans, keeps the current span in a ContextVar, hands finished spans to a
background processor. Contract: nothing in here may raise into, block, or slow down user code
beyond the span bookkeeping itself.
"""
from __future__ import annotations

import atexit
import contextlib
import contextvars
import functools
import inspect
import logging
import threading
import zlib
from collections.abc import AsyncIterator, Callable, Iterator
from types import TracebackType
from typing import Any, TypeVar, overload

from .serialize import DEFAULT_LIMITS, Limits, Redactor, default_redactor, to_jsonable
from .span import NOOP_SPAN, Span, SpanType, new_trace_id

logger = logging.getLogger("evalwright")
F = TypeVar("F", bound=Callable[..., Any])

_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar("evalwright_current_span", default=None)


def current_span() -> Span | None:
    """The innermost open span in this thread / asyncio task, if any."""
    return _current.get()


class _SpanScope:
    """Context manager (sync and async) that opens a span and makes it current."""

    __slots__ = ("_kw", "_token", "_tracer", "span")

    def __init__(self, tracer: Tracer, kw: dict[str, Any]) -> None:
        self._tracer, self._kw = tracer, kw
        self.span: Span | None = None
        self._token: contextvars.Token[Span | None] | None = None

    def __enter__(self) -> Span:
        fields = self._kw
        self.span = self._tracer.start_span(fields.pop("name"), fields.pop("type", "function"),
                                            parent=fields.pop("parent", None))
        if fields:
            self.span.log(**fields)
        self._token = _current.set(self.span)
        return self.span

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None,
                 tb: TracebackType | None) -> None:
        assert self.span is not None
        if ev is not None and not isinstance(ev, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            self.span.set_error(ev)
        if self._token is not None:
            try:
                _current.reset(self._token)
            except ValueError:      # exited in a different context (e.g. a generator moved threads)
                _current.set(None)
        self.span.end()

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, et: type[BaseException] | None, ev: BaseException | None,
                        tb: TracebackType | None) -> None:
        self.__exit__(et, ev, tb)


class Tracer:
    def __init__(self, project: str = "default", processor: Any = None, *, sample_rate: float = 1.0,
                 redactor: Redactor | None = None, limits: Limits = DEFAULT_LIMITS, enabled: bool = True,
                 capture_args: bool = True) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be in [0, 1]")
        self.project = project
        self.processor = processor
        self.sample_rate = sample_rate
        self.redactor = redactor if redactor is not None else default_redactor()
        self.limits = limits
        self.enabled = enabled
        self.capture_args = capture_args
        self._lock = threading.Lock()
        self._n_started = 0
        self._n_serialize_errors = 0

    # ---- spans ------------------------------------------------------------------------
    def _sampled(self, trace_id: str) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        # deterministic per trace: every service sampling by trace id agrees
        return (zlib.crc32(trace_id.encode()) / 0xFFFFFFFF) < self.sample_rate

    def start_span(self, name: str, type: SpanType = "function", parent: Span | None = None) -> Span:
        """Create a span without making it current (call .end() yourself)."""
        self._n_started += 1
        if not self.enabled:
            return NOOP_SPAN
        if parent is None:
            parent = _current.get()
        if parent is not None:
            if not parent.sampled:            # the whole trace was sampled out
                return NOOP_SPAN
            return Span(self, name, type, parent=parent)
        trace_id = new_trace_id()
        if not self._sampled(trace_id):
            return NOOP_SPAN
        return Span(self, name, type, trace_id=trace_id)

    def span(self, name: str, type: SpanType = "function", *, parent: Span | None = None,
             **fields: Any) -> _SpanScope:
        """``with tracer.span("retrieve", type="retrieval", input=q) as s: ...`` (also ``async with``)."""
        return _SpanScope(self, {"name": name, "type": type, "parent": parent, **fields})

    def _on_end(self, span: Span) -> None:
        if self.processor is None:
            return
        try:
            record = self.serialize(span)
            self.processor.on_end(record)
        except Exception:
            self._n_serialize_errors += 1
            logger.debug("evalwright: failed to record span %s", span.name, exc_info=True)

    def serialize(self, span: Span) -> dict[str, Any]:
        d = span.to_dict()
        d["project"] = self.project
        for k in ("input", "output", "expected", "metadata"):
            v = d[k]
            if v is not None and v != {}:
                d[k] = to_jsonable(v, self.limits, self.redactor)
        if d["metrics"]:
            d["metrics"] = to_jsonable(d["metrics"], self.limits)
        if d["scores"]:
            d["scores"] = to_jsonable(d["scores"], self.limits)
        return d

    # ---- decorator ----------------------------------------------------------------------
    @overload
    def traced(self, fn: F) -> F: ...

    @overload
    def traced(self, fn: None = None, *, name: str | None = None, type: SpanType = "function",
               capture_args: bool | None = None, capture_output: bool = True,
               profile: bool | int = False) -> Callable[[F], F]: ...

    def traced(self, fn: F | None = None, *, name: str | None = None, type: SpanType = "function",
               capture_args: bool | None = None, capture_output: bool = True,
               profile: bool | int = False) -> F | Callable[[F], F]:
        """Trace a function. Works on sync, async, generator and async-generator functions.
        profile=True (or an int = top-N functions) attaches a cProfile report to sampled spans."""

        def deco(f: F) -> F:
            return _wrap(self, f, name or f.__qualname__, type,
                         self.capture_args if capture_args is None else capture_args, capture_output, profile)
        return deco(fn) if fn is not None else deco

    # ---- lifecycle ----------------------------------------------------------------------
    def flush(self, timeout: float | None = 5.0) -> bool:
        p = self.processor
        return bool(p.flush(timeout)) if p is not None and hasattr(p, "flush") else True

    def shutdown(self, timeout: float | None = 5.0) -> None:
        p = self.processor
        if p is not None and hasattr(p, "shutdown"):
            p.shutdown(timeout)

    def stats(self) -> dict[str, Any]:
        s: dict[str, Any] = {"spans_started": self._n_started, "serialize_errors": self._n_serialize_errors}
        p = self.processor
        if p is not None and hasattr(p, "stats"):
            s.update(p.stats())
        return s


def _reset(tok: contextvars.Token[Span | None]) -> None:
    try:
        _current.reset(tok)
    except ValueError:
        _current.set(None)


def _binder(fn: Callable[..., Any]) -> Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]:
    """Resolve the signature once at decoration time (inspect.signature is ~10us per call)."""
    try:
        sig: inspect.Signature | None = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    params = list(sig.parameters.values()) if sig is not None else []
    simple = sig is not None and all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params)
    names = [p.name for p in params if p.name not in ("self", "cls")]
    offset = len(params) - len(names)          # skip self/cls positionally

    def bind(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        if simple and len(args) - offset <= len(names):
            out = dict(zip(names, args[offset:], strict=False))   # fewer args than names is valid
            if kwargs:
                out.update(kwargs)
            return out
        if sig is None:
            return {"args": list(args), "kwargs": kwargs}
        try:
            out = dict(sig.bind_partial(*args, **kwargs).arguments)
        except TypeError:
            return {"args": list(args), "kwargs": kwargs}
        out.pop("self", None)
        out.pop("cls", None)
        return out
    return bind


class _GlobalTracer:
    """Resolves the process-global tracer at call time, so functions decorated at import
    (before init()) still report to whatever tracer init() installs later."""

    def span(self, name: str, type: SpanType = "function", **kw: Any) -> _SpanScope:
        return get_tracer().span(name, type, **kw)

    def start_span(self, name: str, type: SpanType = "function", parent: Span | None = None) -> Span:
        return get_tracer().start_span(name, type, parent)


def _wrap(tracer: Tracer | _GlobalTracer, fn: F, name: str, type: SpanType, cap_args: bool,
          cap_out: bool, profile: bool | int = False) -> F:
    bind = _binder(fn) if cap_args else None
    top = 15 if profile is True else int(profile)

    def prof(s: Span) -> contextlib.AbstractContextManager[None]:
        if not top:
            return contextlib.nullcontext()
        from .profiling import profile_into
        return profile_into(s, top)
    if inspect.isasyncgenfunction(fn):
        @functools.wraps(fn)
        async def agen(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            # The span is current only while the generator body runs, never while it is
            # suspended at a yield (otherwise it would leak into the consumer's context).
            s = tracer.start_span(name, type)
            if bind is not None and s.sampled:
                s.log(input=bind(args, kwargs))
            it = fn(*args, **kwargs).__aiter__()
            n = 0
            try:
                while True:
                    tok = _current.set(s)
                    try:
                        item = await it.__anext__()
                    except StopAsyncIteration:
                        break
                    finally:
                        _reset(tok)
                    n += 1
                    yield item
            except BaseException as e:
                if not isinstance(e, (GeneratorExit, KeyboardInterrupt, SystemExit)):
                    s.set_error(e)
                raise
            finally:
                s.log(metrics={"items": n})
                s.end()
        return agen  # type: ignore[return-value]

    if inspect.isgeneratorfunction(fn):
        @functools.wraps(fn)
        def gen(*args: Any, **kwargs: Any) -> Iterator[Any]:
            s = tracer.start_span(name, type)
            if bind is not None and s.sampled:
                s.log(input=bind(args, kwargs))
            it = iter(fn(*args, **kwargs))
            n = 0
            try:
                while True:
                    tok = _current.set(s)
                    try:
                        item = next(it)
                    except StopIteration:
                        break
                    finally:
                        _reset(tok)
                    n += 1
                    yield item
            except BaseException as e:
                if not isinstance(e, (GeneratorExit, KeyboardInterrupt, SystemExit)):
                    s.set_error(e)
                raise
            finally:
                s.log(metrics={"items": n})
                s.end()
        return gen  # type: ignore[return-value]

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def acall(*args: Any, **kwargs: Any) -> Any:
            async with tracer.span(name, type) as s:
                if bind is not None and s.sampled:
                    s.log(input=bind(args, kwargs))
                with prof(s):
                    out = await fn(*args, **kwargs)
                if cap_out and s.sampled:
                    s.log(output=out)
                return out
        return acall  # type: ignore[return-value]

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        with tracer.span(name, type) as s:
            if bind is not None and s.sampled:
                s.log(input=bind(args, kwargs))
            with prof(s):
                out = fn(*args, **kwargs)
            if cap_out and s.sampled:
                s.log(output=out)
            return out
    return call  # type: ignore[return-value]


# ---- process-global tracer -----------------------------------------------------------------
_global: Tracer = Tracer(enabled=False)          # no-op until init(); spans still work, nothing exported
_global_lock = threading.Lock()


def get_tracer() -> Tracer:
    return _global


def set_tracer(t: Tracer) -> Tracer:
    global _global
    with _global_lock:
        old, _global = _global, t
    if old is not t:
        old.shutdown(timeout=1.0)
    return t


def _atexit_flush() -> None:
    with contextlib.suppress(Exception):
        _global.flush(timeout=2.0)


atexit.register(_atexit_flush)


@overload
def traced(fn: F) -> F: ...


@overload
def traced(fn: None = None, *, name: str | None = None, type: SpanType = "function",
           capture_args: bool = True, capture_output: bool = True,
           profile: bool | int = False) -> Callable[[F], F]: ...


def traced(fn: F | None = None, *, name: str | None = None, type: SpanType = "function",
           capture_args: bool = True, capture_output: bool = True,
           profile: bool | int = False) -> F | Callable[[F], F]:
    """Module-level decorator bound to the global tracer (lazily)."""

    def deco(f: F) -> F:
        return _wrap(_GlobalTracer(), f, name or f.__qualname__, type, capture_args, capture_output, profile)
    return deco(fn) if fn is not None else deco


def run_in_thread_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a callable so it runs with the caller's current span (for thread pools)."""
    ctx = contextvars.copy_context()
    return functools.partial(ctx.run, fn)

