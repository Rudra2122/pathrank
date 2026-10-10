"""Exporters. All stdlib (zero runtime dependencies)."""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import threading
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import IO, Any

from ..serialize import dumps
from .base import ExportResult

_RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


class InMemoryExporter:
    """Keeps spans in a list. For tests and for evals that read their own traces back."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def export(self, batch: list[dict[str, Any]]) -> ExportResult:
        with self._lock:
            self.spans.extend(batch)
        return ExportResult.ok()

    def close(self) -> None:
        pass

    def clear(self) -> None:
        with self._lock:
            self.spans.clear()


class JsonlExporter:
    """One JSON object per line. Safe across threads in one process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, batch: list[dict[str, Any]]) -> ExportResult:
        try:
            data = "".join(dumps(r) + "\n" for r in batch)
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(data)
            return ExportResult.ok()
        except OSError as e:
            return ExportResult.retry(f"OSError: {e}")
        except (TypeError, ValueError) as e:
            return ExportResult.fail(f"unserializable batch: {e}")

    def close(self) -> None:
        pass


class ConsoleExporter:
    def __init__(self, stream: IO[str] | None = None) -> None:
        self.stream = stream or sys.stderr

    def export(self, batch: list[dict[str, Any]]) -> ExportResult:
        for r in batch:
            dur = r.get("duration_ms")
            self.stream.write(f"[{r['type']}] {r['name']} {dur:.2f}ms {r['status']}"
                              f"{' parent=' + r['parent_id'][:6] if r.get('parent_id') else ''}\n"
                              if dur is not None else f"[{r['type']}] {r['name']}\n")
        return ExportResult.ok()

    def close(self) -> None:
        pass


def _retry_after(h: str | None) -> float | None:
    if not h:
        return None
    try:
        return float(h)
    except ValueError:
        try:
            from datetime import datetime, timezone
            return max(0.0, (parsedate_to_datetime(h) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


class HttpExporter:
    """POST batches as JSON (optionally gzip) to an ingestion endpoint.

    2xx -> ok; 408/425/429/5xx and network errors -> retryable (Retry-After honored);
    other 4xx -> not retryable (bad payload or auth: retrying won't help).
    """

    def __init__(self, url: str, api_key: str | None = None, *, headers: dict[str, str] | None = None,
                 timeout: float = 10.0, gzip_body: bool = True) -> None:
        self.url, self.timeout, self.gzip_body = url, timeout, gzip_body
        self.headers = {"Content-Type": "application/json", "User-Agent": _user_agent(), **(headers or {})}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def payload(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {"spans": batch}

    def export(self, batch: list[dict[str, Any]]) -> ExportResult:
        body = json.dumps(self.payload(batch), separators=(",", ":")).encode()
        headers = dict(self.headers)
        headers["Idempotency-Key"] = batch_key(batch)     # identical on every retry of this batch
        if self.gzip_body:
            body = gzip.compress(body)
            headers["Content-Encoding"] = "gzip"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
            return ExportResult.ok()
        except urllib.error.HTTPError as e:
            if e.code in _RETRYABLE:
                return ExportResult.retry(f"HTTP {e.code}", _retry_after(e.headers.get("Retry-After")))
            return ExportResult.fail(f"HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            return ExportResult.retry(f"{type(e).__name__}: {e}")

    def close(self) -> None:
        pass


class OtlpHttpExporter(HttpExporter):
    """OpenTelemetry OTLP/HTTP JSON (e.g. http://localhost:4318/v1/traces), so traces land in any
    OTel collector, Jaeger, Grafana Tempo, etc. Span fields map to `evalwright.*` attributes."""

    def __init__(self, endpoint: str = "http://localhost:4318/v1/traces", service_name: str = "evalwright",
                 **kw: Any) -> None:
        super().__init__(endpoint, **kw)
        self.service_name = service_name

    def payload(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return to_otlp(batch, self.service_name)


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        v: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": str(value)}
    elif isinstance(value, float):
        v = {"doubleValue": value}
    elif isinstance(value, str):
        v = {"stringValue": value}
    else:
        v = {"stringValue": json.dumps(value, separators=(",", ":"), default=str)}
    return {"key": key, "value": v}


def to_otlp(batch: list[dict[str, Any]], service_name: str) -> dict[str, Any]:
    spans = []
    for r in batch:
        attrs = [_attr("evalwright.type", r["type"]), _attr("evalwright.project", r.get("project", ""))]
        for k in ("input", "output", "expected"):
            if r.get(k) is not None:
                attrs.append(_attr(f"evalwright.{k}", r[k]))
        for k, v in (r.get("metadata") or {}).items():
            attrs.append(_attr(f"evalwright.metadata.{k}", v))
        for k, v in (r.get("metrics") or {}).items():
            attrs.append(_attr(f"evalwright.metrics.{k}", v))
        for k, v in (r.get("scores") or {}).items():
            if v is not None:
                attrs.append(_attr(f"evalwright.scores.{k}", float(v)))
        s: dict[str, Any] = {"traceId": r["trace_id"], "spanId": r["span_id"], "name": r["name"],
                             "kind": 3 if r["type"] in ("llm", "http") else 1,
                             "startTimeUnixNano": str(r["start_ns"]), "endTimeUnixNano": str(r["end_ns"]),
                             "attributes": attrs,
                             "status": {"code": 2, "message": (r.get("error") or {}).get("message", "")}
                             if r["status"] == "error" else {"code": 1}}
        if r.get("parent_id"):
            s["parentSpanId"] = r["parent_id"]
        spans.append(s)
    return {"resourceSpans": [{"resource": {"attributes": [_attr("service.name", service_name)]},
                               "scopeSpans": [{"scope": {"name": "evalwright", "version": _version()},
                                               "spans": spans}]}]}


def _version() -> str:
    from .._version import __version__
    return __version__


def batch_key(batch: list[dict[str, Any]]) -> str:
    """Content-derived idempotency key (wire protocol v1): sha256 of span ids joined by newlines."""
    ids = "\n".join(str(r.get("span_id", "")) for r in batch)
    return hashlib.sha256(ids.encode()).hexdigest()[:32]


def _user_agent() -> str:
    return f"evalwright-python/{_version()} python/{sys.version_info.major}.{sys.version_info.minor}"
