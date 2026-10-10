"""Wire protocol v1: schema conformance, idempotency, and delivery through injected failures."""
from __future__ import annotations

import json

import pytest

import evalwright as ew
from evalwright.devserver import serve_in_thread, validate_record
from evalwright.export.exporters import batch_key


def _records(n: int = 3) -> list[dict]:
    exp = ew.InMemoryExporter()
    t = ew.init("wire", exporter=exp, flush_interval=0.01)
    for i in range(n):
        with ew.span(f"op{i}", type="llm", input={"q": i}) as s:
            s.log(output="x", scores={"acc": 1.0}, metrics={"tokens": 5}, tags=["a"])
    t.flush(5)
    return list(exp.spans)


def test_python_records_match_schema_and_validator(span_schema):
    recs = _records()
    assert recs and all(validate_record(r) == [] for r in recs)
    jsonschema = pytest.importorskip("jsonschema")
    for r in recs:
        jsonschema.validate(r, span_schema)


def test_validator_rejects_bad_records():
    good = _records(1)[0]
    assert "trace_id must be 32 lowercase hex" in validate_record({**good, "trace_id": "XYZ"})
    assert "scores must map to null or numbers in [0, 1]" in validate_record({**good, "scores": {"a": 3}})
    assert any("unexpected field" in e for e in validate_record({**good, "extra": 1}))
    assert any("missing" in e for e in validate_record({k: v for k, v in good.items() if k != "project"}))


def test_batch_key_is_content_derived():
    recs = _records(2)
    assert batch_key(recs) == batch_key([dict(r) for r in recs])   # same spans -> same key (retries dedupe)
    assert batch_key(recs) != batch_key(recs[:1])
    assert len(batch_key(recs)) == 32


def test_duplicate_batch_is_deduped_by_collector():
    srv, col, url = serve_in_thread()
    try:
        exp = ew.HttpExporter(f"{url}/v1/spans")
        recs = _records(2)
        assert exp.export(recs).success and exp.export(recs).success   # second = a retry that already landed
        assert col.counts["accepted_batches"] == 1 and col.counts["duplicate_batches"] == 1
        assert len(col.spans) == 2
    finally:
        srv.shutdown()


def test_every_span_delivered_exactly_once_through_failures():
    """40% of requests fail with 503 + Retry-After; retries must deliver everything, nothing twice."""
    srv, col, url = serve_in_thread(fail_rate=0.4, fail_status=503, retry_after=0.01, seed=7)
    try:
        t = ew.init("wire", exporter=ew.HttpExporter(f"{url}/v1/spans"), flush_interval=0.01, max_batch=7,
                    retry=ew.RetryPolicy(max_retries=20, base_delay=0.001, max_delay=0.02))
        for i in range(100):
            with ew.span("op", input=i):
                pass
        assert t.flush(30)
        assert col.counts["injected_failures"] > 0
        ids = [s["span_id"] for s in col.spans]
        assert len(ids) == 100 and len(set(ids)) == 100
        assert all(validate_record(s) == [] for s in col.spans)
        assert any(ua.startswith("evalwright-python/") for ua in col.user_agents)
    finally:
        srv.shutdown()


def test_non_retryable_4xx_is_dropped_not_retried():
    srv, col, url = serve_in_thread(api_key="right")
    try:
        t = ew.init("wire", exporter=ew.HttpExporter(f"{url}/v1/spans", api_key="wrong"), flush_interval=0.01,
                    retry=ew.RetryPolicy(max_retries=5, base_delay=0.001))
        with ew.span("op"):
            pass
        t.flush(5)
        assert col.counts["requests"] == 1          # 401: one attempt, no retries
        assert col.spans == []
    finally:
        srv.shutdown()


@pytest.mark.parametrize("patch,problem", [
    ({"span_id": "abc"}, "span_id must be 16 lowercase hex"),
    ({"parent_id": "zz"}, "parent_id must be null or 16 lowercase hex"),
    ({"name": ""}, "name must be a non-empty string"),
    ({"type": "weird"}, "type 'weird' not allowed"),
    ({"start_ns": -1}, "start_ns must be a non-negative integer"),
    ({"end_ns": True}, "end_ns must be a non-negative integer"),
    ({"duration_ms": "1"}, "duration_ms must be a non-negative number"),
    ({"cpu_ms": -2}, "cpu_ms must be null or a non-negative number"),
    ({"metadata": []}, "metadata must be an object"),
    ({"metrics": {"t": "x"}}, "metrics must map to numbers"),
    ({"tags": ["a", "a"]}, "tags must be unique strings"),
    ({"status": "meh"}, "status must be ok|error"),
    ({"error": {"type": "E"}}, "error must be null or {type, message, stack} strings"),
    ({"project": ""}, "project must be a non-empty string"),
])
def test_validator_rule(patch, problem):
    rec = {**_records(1)[0], **patch}
    assert problem in validate_record(rec)


def test_validator_non_object():
    assert validate_record([]) == ["record is not an object"]


def test_collector_http_surface():
    import gzip
    import urllib.error
    import urllib.request

    srv, _col, url = serve_in_thread()
    try:
        def post(body: bytes, headers: dict) -> int:
            req = urllib.request.Request(f"{url}/v1/spans", data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code

        assert post(b"not json", {}) == 400
        assert post(json.dumps({"spans": [{"bad": 1}]}).encode(), {}) == 400
        ok = gzip.compress(json.dumps({"spans": _records(1)}).encode())
        assert post(ok, {"Content-Encoding": "gzip"}) == 200
        with urllib.request.urlopen(f"{url}/v1/stats") as r:
            stats = json.loads(r.read())
        assert stats["invalid_batches"] == 2 and stats["spans"] == 1
        with urllib.request.urlopen(f"{url}/health") as r:
            assert r.status == 200
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{url}/nope")
    finally:
        srv.shutdown()


def test_collector_writes_jsonl(tmp_path):
    out = tmp_path / "spans.jsonl"
    srv, _col, url = serve_in_thread(out=str(out))
    try:
        assert ew.HttpExporter(f"{url}/v1/spans").export(_records(3)).success
        assert len(out.read_text().splitlines()) == 3
    finally:
        srv.shutdown()
