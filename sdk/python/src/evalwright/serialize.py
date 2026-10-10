"""Turn arbitrary user objects into bounded, JSON-safe data. Never raises.

Logged inputs/outputs come from user code: model responses, numpy arrays, pydantic models,
exceptions, objects with cycles, megabyte strings. Tracing must not crash or bloat because of
them, so everything goes through :func:`to_jsonable`, which is total (always returns) and
bounded (depth, string length, collection size).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import functools
import json
import math
import re
from collections.abc import Callable, Mapping
from pathlib import PurePath
from typing import Any

Redactor = Callable[[str, Any], Any]

_SECRET_KEY = re.compile(r"(pass(word)?|secret|token|api[_-]?key|authorization|auth|cookie|session|private[_-]?key)",
                         re.IGNORECASE)
_SECRET_VALUE = re.compile(r"(sk-[A-Za-z0-9_\-]{16,}|Bearer\s+[A-Za-z0-9._\-]{16,}|AKIA[0-9A-Z]{16})")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
REDACTED = "[REDACTED]"


@dataclasses.dataclass(frozen=True)
class Limits:
    max_depth: int = 8
    max_string: int = 10_000
    max_items: int = 200


DEFAULT_LIMITS = Limits()


@functools.lru_cache(maxsize=8192)
def _is_secret_key(key: str) -> bool:      # keys repeat constantly; the regex result is cached
    return _SECRET_KEY.search(key) is not None


def default_redactor(redact_emails: bool = False) -> Redactor:
    """Mask values under secret-looking keys and secret-looking strings anywhere."""

    def redact(key: str, value: Any) -> Any:
        if key and _is_secret_key(key) and value not in (None, "", 0, False):
            return REDACTED
        if isinstance(value, str):
            value = _SECRET_VALUE.sub(REDACTED, value)
            if redact_emails:
                value = _EMAIL.sub("[EMAIL]", value)
        return value

    return redact


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"...[truncated {len(s) - n} chars]"


_SCALARS = (bool, int, float, type(None))


def to_jsonable(obj: Any, limits: Limits = DEFAULT_LIMITS, redactor: Redactor | None = None,
                _key: str = "", _depth: int = 0, _seen: set[int] | None = None) -> Any:
    # fast path: plain scalars and short strings need no walk (most logged values)
    t = type(obj)
    if t in _SCALARS and redactor is None and (t is not float or math.isfinite(obj)) \
            and (t is not int or -(2 ** 63) <= obj < 2 ** 63):
        return obj
    try:
        return _convert(obj, limits, redactor, _key, _depth, _seen if _seen is not None else set())
    except Exception as e:  # noqa: BLE001 - total by design
        return f"<unserializable {type(obj).__name__}: {type(e).__name__}>"


def _convert(obj: Any, lim: Limits, red: Redactor | None, key: str, depth: int, seen: set[int]) -> Any:
    if red is not None:
        obj = red(key, obj)
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj if -(2 ** 63) <= obj < 2 ** 63 else str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, str):
        return _truncate(obj, lim.max_string)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, enum.Enum):
        return _convert(obj.value, lim, None, key, depth, seen)
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, _dt.timedelta):
        return obj.total_seconds()
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, BaseException):
        return {"type": type(obj).__name__, "message": _truncate(str(obj), lim.max_string)}
    if depth >= lim.max_depth:
        return f"<max depth: {type(obj).__name__}>"

    oid = id(obj)
    if oid in seen:
        return f"<cycle: {type(obj).__name__}>"
    seen.add(oid)
    try:
        nxt = depth + 1
        if isinstance(obj, Mapping):
            out: dict[str, Any] = {}
            for i, (k, v) in enumerate(obj.items()):
                if i >= lim.max_items:
                    out["..."] = f"{len(obj) - lim.max_items} more keys"
                    break
                sk = k if isinstance(k, str) else str(k)
                out[sk] = _convert(v, lim, red, sk, nxt, seen)
            return out
        if isinstance(obj, (list, tuple, set, frozenset)):
            seq = list(obj) if not isinstance(obj, (list, tuple)) else obj
            items = [_convert(v, lim, red, key, nxt, seen) for v in seq[: lim.max_items]]
            if len(seq) > lim.max_items:
                items.append(f"...{len(seq) - lim.max_items} more items")
            return items
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: _convert(getattr(obj, f.name), lim, red, f.name, nxt, seen)
                    for f in dataclasses.fields(obj)[: lim.max_items]}
        # numpy arrays/scalars, pandas objects, torch tensors: duck-typed, no imports
        if hasattr(obj, "tolist") and hasattr(obj, "dtype"):
            size = getattr(obj, "size", None)
            if isinstance(size, int) and size > lim.max_items:
                try:
                    head: Any = _convert(obj.reshape(-1)[: lim.max_items].tolist(), lim, red, key, nxt, seen)
                except Exception:  # noqa: BLE001 - exotic array type: keep the summary, skip the head
                    head = None
                return {"type": type(obj).__name__, "shape": list(getattr(obj, "shape", ())),
                        "dtype": str(obj.dtype), "head": head}
            return _convert(obj.tolist(), lim, red, key, nxt, seen)
        for meth in ("model_dump", "dict", "to_dict", "_asdict"):   # pydantic v2/v1, pandas, namedtuple
            f = getattr(obj, meth, None)
            if callable(f):
                try:
                    return _convert(f(), lim, red, key, nxt, seen)
                except TypeError:
                    continue
        if hasattr(obj, "__dict__") and not callable(obj):
            d = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
            return {"__type__": type(obj).__name__, **_convert(d, lim, red, key, nxt, seen)}
        return _truncate(repr(obj), min(lim.max_string, 500))
    finally:
        seen.discard(oid)


def dumps(obj: Any) -> str:
    """Compact JSON for already-jsonable data."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
