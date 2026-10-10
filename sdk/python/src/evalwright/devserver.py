"""Local collector implementing wire protocol v1 (sdk/spec/PROTOCOL.md).

For development and tests of any evalwright SDK:
  * validates every record against the v1 schema (stdlib-only checker, no jsonschema needed)
  * dedupes retried batches by Idempotency-Key
  * fault injection: --fail-rate / --fail-status / --retry-after to exercise client retries
  * GET /v1/spans returns everything accepted; GET /v1/stats returns counters

    evalwright serve --port 8787 --fail-rate 0.3
"""
from __future__ import annotations

import gzip
import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_TYPES = {"task", "llm", "tool", "function", "eval", "score", "http", "retrieval", "custom"}
_REQUIRED = ("trace_id", "span_id", "parent_id", "name", "type", "start_ns", "end_ns", "duration_ms", "cpu_ms",
             "input", "output", "expected", "metadata", "metrics", "scores", "tags", "status", "error", "project")


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_record(r: Any) -> list[str]:
    """Problems with one record (empty list = valid). Mirrors sdk/spec/span.schema.json."""
    if not isinstance(r, dict):
        return ["record is not an object"]
    errs = [f"missing {k}" for k in _REQUIRED if k not in r]
    errs += [f"unexpected field {k}" for k in r if k not in _REQUIRED]
    if errs:
        return errs
    if not (isinstance(r["trace_id"], str) and _HEX32.match(r["trace_id"])):
        errs.append("trace_id must be 32 lowercase hex")
    if not (isinstance(r["span_id"], str) and _HEX16.match(r["span_id"])):
        errs.append("span_id must be 16 lowercase hex")
    if r["parent_id"] is not None and not (isinstance(r["parent_id"], str) and _HEX16.match(r["parent_id"])):
        errs.append("parent_id must be null or 16 lowercase hex")
    if not (isinstance(r["name"], str) and r["name"]):
        errs.append("name must be a non-empty string")
    if r["type"] not in _TYPES:
        errs.append(f"type {r['type']!r} not allowed")
    for k in ("start_ns", "end_ns"):
        if not (isinstance(r[k], int) and not isinstance(r[k], bool) and r[k] >= 0):
            errs.append(f"{k} must be a non-negative integer")
    if not (_num(r["duration_ms"]) and r["duration_ms"] >= 0):
        errs.append("duration_ms must be a non-negative number")
    if r["cpu_ms"] is not None and not (_num(r["cpu_ms"]) and r["cpu_ms"] >= 0):
        errs.append("cpu_ms must be null or a non-negative number")
    if not isinstance(r["metadata"], dict):
        errs.append("metadata must be an object")
    if not (isinstance(r["metrics"], dict) and all(_num(v) for v in r["metrics"].values())):
        errs.append("metrics must map to numbers")
    if not (isinstance(r["scores"], dict)
            and all(v is None or (_num(v) and 0 <= v <= 1) for v in r["scores"].values())):
        errs.append("scores must map to null or numbers in [0, 1]")
    tags = r["tags"]
    if not (isinstance(tags, list) and all(isinstance(t, str) for t in tags) and len(set(tags)) == len(tags)):
        errs.append("tags must be unique strings")
    if r["status"] not in ("ok", "error"):
        errs.append("status must be ok|error")
    e = r["error"]
    if e is not None and not (isinstance(e, dict) and set(e) == {"type", "message", "stack"}
                              and all(isinstance(v, str) for v in e.values())):
        errs.append("error must be null or {type, message, stack} strings")
    if not (isinstance(r["project"], str) and r["project"]):
        errs.append("project must be a non-empty string")
    return errs


class Collector:
    def __init__(self, fail_rate: float = 0.0, fail_status: int = 503, retry_after: float | None = None,
                 api_key: str | None = None, out: str | None = None, seed: int = 0) -> None:
        self.fail_rate, self.fail_status, self.retry_after = fail_rate, fail_status, retry_after
        self.api_key, self.out = api_key, out
        self.spans: list[dict[str, Any]] = []
        self.seen_keys: set[str] = set()
        self.counts: dict[str, int] = dict.fromkeys(
            ("requests", "accepted_batches", "duplicate_batches", "injected_failures", "invalid_batches", "spans"), 0)
        self.user_agents: dict[str, int] = {}
        self._lock = threading.Lock()
        self._rng = random.Random(seed)

    @property
    def stats(self) -> dict[str, Any]:
        return {**self.counts, "user_agents": dict(self.user_agents)}

    def handle(self, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any], dict[str, str]]:
        with self._lock:
            self.counts["requests"] += 1
            ua = headers.get("user-agent", "")
            self.user_agents[ua] = self.user_agents.get(ua, 0) + 1
            if self.api_key and headers.get("authorization") != f"Bearer {self.api_key}":
                return 401, {"error": "unauthorized"}, {}
            if self.fail_rate and self._rng.random() < self.fail_rate:
                self.counts["injected_failures"] += 1
                extra = {"Retry-After": str(self.retry_after)} if self.retry_after is not None else {}
                return self.fail_status, {"error": "injected failure"}, extra
            try:
                if headers.get("content-encoding") == "gzip":
                    body = gzip.decompress(body)
                payload = json.loads(body)
                spans = payload["spans"]
                assert isinstance(spans, list)
            except Exception:  # noqa: BLE001
                self.counts["invalid_batches"] += 1
                return 400, {"error": "body must be {\"spans\": [...]} JSON"}, {}
            problems = {i: p for i, s in enumerate(spans) if (p := validate_record(s))}
            if problems:
                self.counts["invalid_batches"] += 1
                return 400, {"error": "schema", "problems": {str(k): v for k, v in list(problems.items())[:5]}}, {}
            key = headers.get("idempotency-key")
            if key and key in self.seen_keys:
                self.counts["duplicate_batches"] += 1
                return 200, {"accepted": 0, "duplicate": True}, {}
            if key:
                self.seen_keys.add(key)
            self.spans.extend(spans)
            self.counts["accepted_batches"] += 1
            self.counts["spans"] += len(spans)
            if self.out:
                with open(self.out, "a", encoding="utf-8") as f:
                    for s in spans:
                        f.write(json.dumps(s, separators=(",", ":")) + "\n")
            return 200, {"accepted": len(spans)}, {}


def make_server(host: str = "127.0.0.1", port: int = 0, **kw: Any) -> tuple[ThreadingHTTPServer, Collector]:
    col = Collector(**kw)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, code: int, obj: Any, extra: dict[str, str] | None = None) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.rstrip("/") != "/v1/spans":
                return self._send(404, {"error": "not found"})
            code, obj, extra = col.handle({k.lower(): v for k, v in self.headers.items()}, body)
            self._send(code, obj, extra)

        def do_GET(self) -> None:
            p = self.path.split("?")[0].rstrip("/")
            if p == "/v1/spans":
                with col._lock:
                    return self._send(200, {"spans": list(col.spans)})
            if p == "/v1/stats":
                with col._lock:
                    return self._send(200, col.stats)
            if p in ("", "/health"):
                return self._send(200, {"status": "ok"})
            self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv, col


def serve_in_thread(**kw: Any) -> tuple[ThreadingHTTPServer, Collector, str]:
    """Start a collector on a free port in a daemon thread. Returns (server, collector, base_url)."""
    srv, col = make_server(**kw)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    return srv, col, f"http://{kw.get('host', '127.0.0.1')}:{port}"
