from __future__ import annotations

import gzip
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import evalwright as ew
from evalwright.export import BatchProcessor, CircuitBreaker, ExportResult, RetryPolicy, to_otlp

FAST = RetryPolicy(max_retries=3, base_delay=0.001, max_delay=0.01)


def rec(i: int) -> dict:
    return {"trace_id": "a" * 32, "span_id": f"{i:016x}", "parent_id": None, "name": f"s{i}", "type": "task",
            "start_ns": 1, "end_ns": 2, "status": "ok", "duration_ms": 1.0}


class Scripted:
    """Exporter returning a scripted sequence of results, then ok."""

    def __init__(self, script=()):
        self.script, self.calls, self.got = list(script), 0, []
        self.lock = threading.Lock()

    def export(self, batch):
        with self.lock:
            self.calls += 1
            r = self.script.pop(0) if self.script else ExportResult.ok()
            if isinstance(r, Exception):
                raise r
            if r.success:
                self.got.extend(batch)
            return r

    def close(self):
        pass


def test_batches_and_flush():
    exp = Scripted()
    p = BatchProcessor(exp, max_batch=10, flush_interval=0.01, retry=FAST)
    for i in range(35):
        p.on_end(rec(i))
    assert p.flush(2.0)
    assert len(exp.got) == 35 and exp.calls >= 4
    st = p.stats()
    assert st["exported"] == 35 and st["queued"] == 0 and st["circuit"] == "closed"
    p.shutdown(1.0)


def test_queue_overflow_drops_oldest_without_blocking():
    gate = threading.Event()

    class Slow(Scripted):
        def export(self, batch):
            gate.wait(2)
            return super().export(batch)

    exp = Slow()
    p = BatchProcessor(exp, max_queue=5, max_batch=1, flush_interval=0.01, retry=FAST)
    t0 = time.perf_counter()
    for i in range(50):
        p.on_end(rec(i))
    assert time.perf_counter() - t0 < 0.5                       # producer never blocked
    gate.set()
    p.flush(2.0)
    st = p.stats()
    assert st["dropped_queue_full"] >= 40
    assert exp.got[-1]["name"] == "s49"                          # newest survived
    p.shutdown(1.0)


def test_retry_then_success_and_exception_tolerance():
    exp = Scripted([ExportResult.retry("503"), RuntimeError("exporter bug"), ExportResult.retry("429", 0.001)])
    p = BatchProcessor(exp, max_batch=100, flush_interval=0.01, retry=FAST)
    for i in range(3):
        p.on_end(rec(i))
    assert p.flush(2.0)
    st = p.stats()
    assert len(exp.got) == 3 and st["retries"] == 3 and st["dropped_failed"] == 0
    p.shutdown(1.0)


def test_non_retryable_drops_batch_and_counts():
    exp = Scripted([ExportResult.fail("HTTP 400")])
    p = BatchProcessor(exp, max_batch=100, flush_interval=0.01, retry=FAST)
    p.on_end(rec(0))
    assert p.flush(2.0)
    st = p.stats()
    assert st["dropped_failed"] == 1 and st["last_error"] == "HTTP 400" and st["retries"] == 0
    p.shutdown(1.0)


def test_retry_policy_honors_retry_after_and_caps():
    import random
    pol = RetryPolicy(base_delay=1.0, max_delay=5.0)
    rng = random.Random(0)
    assert pol.delay(0, 2.5, rng) == 2.5
    assert pol.delay(0, 100.0, rng) == 5.0
    assert all(0 <= pol.delay(10, None, rng) <= 5.0 for _ in range(100))


def test_circuit_breaker_open_half_open_close():
    now = [0.0]
    cb = CircuitBreaker(threshold=3, cooldown=10.0, clock=lambda: now[0])
    for _ in range(3):
        cb.record(False)
    assert cb.state == "open" and not cb.allow()
    now[0] = 10.5
    assert cb.state == "half_open" and cb.allow()
    cb.record(False)                                  # probe fails -> re-open
    assert cb.state == "open"
    now[0] = 21.0
    cb.record(True)
    assert cb.state == "closed" and cb.failures == 0


def test_processor_stops_calling_exporter_while_circuit_open():
    exp = Scripted([ExportResult.fail("x")] * 50)
    p = BatchProcessor(exp, max_batch=1, flush_interval=0.01, retry=RetryPolicy(max_retries=0),
                       breaker=CircuitBreaker(threshold=2, cooldown=60))
    for i in range(10):
        p.on_end(rec(i))
    time.sleep(0.3)
    assert exp.calls == 2 and p.stats()["circuit"] == "open" and p.stats()["queued"] == 8
    assert p.flush(0.1) is False                      # can't complete while open
    p.shutdown(0.1)


def test_flush_times_out_on_hung_exporter():
    hang = threading.Event()

    class Hung(Scripted):
        def export(self, batch):
            hang.wait(5)
            return ExportResult.ok()

    p = BatchProcessor(Hung(), flush_interval=0.01)
    p.on_end(rec(0))
    t0 = time.perf_counter()
    assert p.flush(0.2) is False
    assert time.perf_counter() - t0 < 1.0
    hang.set()
    p.shutdown(1.0)


def test_shutdown_stops_accepting_and_validates_args():
    exp = Scripted()
    p = BatchProcessor(exp, flush_interval=0.01)
    p.on_end(rec(0))
    p.shutdown(1.0)
    p.on_end(rec(1))
    assert [r["name"] for r in exp.got] == ["s0"]
    with pytest.raises(ValueError):
        BatchProcessor(exp, max_batch=0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
@pytest.mark.filterwarnings("ignore:This process.*is multi-threaded:DeprecationWarning")
def test_fork_child_gets_fresh_pipeline(tmp_path):
    out = tmp_path / "child.jsonl"
    exp = ew.InMemoryExporter()
    t = ew.init("fork", exporter=exp, flush_interval=0.01)
    with t.span("parent_before"):
        pass
    t.flush()
    pid = os.fork()
    if pid == 0:                                       # child
        code = 1
        try:
            p = t.processor
            assert p.stats()["queued"] == 0
            p.exporter = ew.JsonlExporter(out)
            with t.span("in_child"):
                pass
            code = 0 if t.flush(2.0) else 3
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert '"name":"in_child"' in out.read_text()
    with t.span("parent_after"):
        pass
    t.flush()
    assert {"parent_before", "parent_after"} <= {s["name"] for s in exp.spans}


# ---- real HTTP fault injection ---------------------------------------------------------------
class _Srv:
    def __init__(self, script):
        self.script, self.requests = list(script), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                outer.requests.append({"headers": dict(self.headers), "json": json.loads(body)})
                code, hdrs = outer.script.pop(0) if outer.script else (200, {})
                self.send_response(code)
                for k, v in hdrs.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/ingest"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def test_http_exporter_status_mapping_and_payload():
    srv = _Srv([(500, {}), (429, {"Retry-After": "0"}), (200, {}), (400, {}), (401, {})])
    try:
        e = ew.HttpExporter(srv.url, api_key="k123")
        assert e.export([rec(0)]).retryable
        r = e.export([rec(0)])
        assert r.retryable and r.retry_after == 0.0
        assert e.export([rec(0)]).success
        assert not e.export([rec(0)]).retryable
        assert not e.export([rec(0)]).success
        h = srv.requests[2]["headers"]
        assert h["Authorization"] == "Bearer k123" and h["Content-Encoding"] == "gzip"
        assert h["User-Agent"].startswith("evalwright-python/")
        assert srv.requests[2]["json"]["spans"][0]["name"] == "s0"
    finally:
        srv.close()


def test_end_to_end_tracer_over_flaky_http():
    srv = _Srv([(503, {}), (502, {})])
    try:
        t = ew.init("flaky", exporter=ew.HttpExporter(srv.url, gzip_body=False), flush_interval=0.01,
                    retry=FAST)
        for i in range(20):
            with t.span(f"op{i}"):
                pass
        assert t.flush(3.0)
        st = t.stats()
        assert st["exported"] == 20 and st["retries"] == 2
        names = {s["name"] for r in srv.requests for s in r["json"]["spans"]}
        assert {f"op{i}" for i in range(20)} <= names
    finally:
        srv.close()


def test_connection_refused_is_retryable():
    r = ew.HttpExporter("http://127.0.0.1:9/never", timeout=0.5).export([rec(0)])
    assert not r.success and r.retryable


def test_retry_after_http_date():
    from email.utils import formatdate

    from evalwright.export.exporters import _retry_after
    assert 0 <= _retry_after(formatdate(time.time() + 3, usegmt=True)) <= 4
    assert _retry_after("garbage") is None and _retry_after(None) is None


def test_otlp_payload_shape():
    t = ew.init("otlp", exporter=(mem := ew.InMemoryExporter()), flush_interval=0.01)
    with t.span("root", type="llm", input="q") as s:
        s.log(scores={"acc": 1.0}, metrics={"tok": 3}, metadata={"model": "m"})
        with pytest.raises(RuntimeError), t.span("child"):
            raise RuntimeError("bad")
    t.flush()
    payload = to_otlp(mem.spans, "svc")
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    by = {s["name"]: s for s in spans}
    assert by["child"]["parentSpanId"] == by["root"]["spanId"] and len(by["root"]["traceId"]) == 32
    assert by["child"]["status"]["code"] == 2 and by["root"]["status"]["code"] == 1 and by["root"]["kind"] == 3
    keys = {a["key"] for a in by["root"]["attributes"]}
    assert {"evalwright.scores.acc", "evalwright.metrics.tok", "evalwright.metadata.model"} <= keys
    json.dumps(payload)
    srv = _Srv([])
    try:
        assert ew.OtlpHttpExporter(srv.url, service_name="svc").export(mem.spans).success
        assert "resourceSpans" in srv.requests[0]["json"]
    finally:
        srv.close()


def test_jsonl_and_console_exporters(tmp_path, capsys):
    j = ew.JsonlExporter(tmp_path / "x" / "t.jsonl")
    assert j.export([rec(0), rec(1)]).success
    assert len((tmp_path / "x" / "t.jsonl").read_text().splitlines()) == 2
    bad = {"x": float("nan")}
    assert not j.export([bad]).retryable
    c = ew.ConsoleExporter(sys.stdout)
    c.export([rec(0), {"type": "t", "name": "n", "duration_ms": None, "status": "ok"}])
    assert "[task] s0" in capsys.readouterr().out
    m = ew.InMemoryExporter()
    m.export([rec(0)])
    m.clear()
    assert m.spans == []
