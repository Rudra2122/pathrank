"""Cross-language contract: the TypeScript SDK (built package) and the Python SDK both speak wire
protocol v1 to the same collector, produce schema-valid records, and share experiment storage.
Skipped when Node or the TS build is unavailable (the sdk-contract CI job always has both)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import evalwright as ew
from evalwright.cli import main as cli
from evalwright.devserver import serve_in_thread, validate_record

SDK = Path(__file__).resolve().parents[2]
TS = SDK / "typescript"
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (TS / "dist" / "index.js").exists(),
    reason="needs node and a built TypeScript SDK (npm run build in sdk/typescript)",
)


def run_ts(url: str, store: Path) -> dict:
    env = {**os.environ, "EVALWRIGHT_URL": url, "EVALWRIGHT_STORE": str(store)}
    p = subprocess.run(["node", "scripts/contract.mjs"], cwd=TS, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_both_sdks_speak_the_same_protocol(tmp_path, span_schema):
    jsonschema = pytest.importorskip("jsonschema")
    srv, col, url = serve_in_thread(fail_rate=0.3, fail_status=503, seed=3)
    try:
        out = run_ts(url, tmp_path / "store")
        assert out["flushed"] and out["stats"]["dropped_failed"] == 0 and out["stats"]["retries"] > 0

        t = ew.init("contract-py", exporter=ew.HttpExporter(f"{url}/v1/spans"), flush_interval=0.01,
                    retry=ew.RetryPolicy(max_retries=25, base_delay=0.001, max_delay=0.02))
        with ew.span("py-root", type="task", input={"q": 1}) as s:
            with ew.span("py-child", type="llm"):
                pass
            s.log(scores={"ok": 1.0})
        assert t.flush(20)

        spans = col.spans
        by_project = {p: [s for s in spans if s["project"] == p] for p in ("contract-ts", "contract-py")}
        assert all(by_project.values())
        for s in spans:                               # every record from both languages is spec-valid
            jsonschema.validate(s, span_schema)
            assert validate_record(s) == []
        assert len({s["span_id"] for s in spans}) == len(spans)        # retries never duplicated anything

        ts = by_project["contract-ts"]
        ids = {s["span_id"]: s for s in ts}
        for s in ts:                                  # parent links resolve within the TS traces
            if s["parent_id"] is not None and s["name"] != "ts-contract-eval":
                assert ids[s["parent_id"]]["trace_id"] == s["trace_id"]
        retrieve = next(s for s in ts if s["name"] == "retrieve")
        assert retrieve["output"]["api_key"] == "[REDACTED]"          # TS redaction == Python redaction
        fail = next(s for s in ts if s["name"] == "failing")
        assert fail["status"] == "error" and fail["error"]["type"] == "TypeError"
        uas = col.user_agents
        assert any(u.startswith("evalwright-typescript/") for u in uas)
        assert any(u.startswith("evalwright-python/") for u in uas)
    finally:
        srv.shutdown()


def test_python_cli_compares_typescript_experiments(tmp_path, capsys):
    srv, _, url = serve_in_thread()
    try:
        store = tmp_path / "store"
        run_ts(url, store)
    finally:
        srv.shutdown()
    exp = ew.load_experiment(store, "ts-contract-eval")
    assert exp is not None and len(exp["cases"]) == 8
    prev = ew.load_experiment(store, "ts-contract-eval", "previous")
    cmp = ew.compare_experiments(prev, exp)
    assert cmp["scores"]["exact_match"]["regressed"] == 4
    assert cli(["compare", "ts-contract-eval", "--store-dir", str(store)]) == 0
    assert "exact_match" in capsys.readouterr().out
    # case ids agree across languages: Python computes the same ids for the same inputs
    from evalwright.evals.types import EvalCase
    assert {c["case_id"] for c in exp["cases"]} == {EvalCase(i, i * 2).case_id() for i in range(8)}
