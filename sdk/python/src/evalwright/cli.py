"""Command line: run eval files, compare experiments, gate CI on regressions.

  evalwright run evals/                       # every *_eval.py / eval_*.py under the path
  evalwright run my_eval.py --max-regression 0.02 --min exact_match=0.8
  evalwright compare qa-bot --baseline previous
  evalwright ls qa-bot

Exit codes: 0 pass, 1 a gate failed, 2 an eval file failed to import or run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

from ._version import __version__
from .evals import core
from .evals.experiment import compare_experiments, list_experiments, load_experiment


def _discover(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(x for x in p.rglob("*.py") if x.name.endswith("_eval.py") or x.name.startswith("eval_"))
        elif p.suffix == ".py":
            out.append(p)
        else:
            raise SystemExit(f"not a .py file or directory: {p}")
    return out


def _load(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"_evalwright_user_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent.resolve()))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)


def _parse_min(items: list[str]) -> dict[str, float]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not v:
            raise SystemExit(f"--min expects name=value, got {it!r}")
        out[k] = float(v)
    return out


def cmd_run(a: argparse.Namespace) -> int:
    core._cli_options.update({"store_dir": a.store_dir, "max_concurrency": a.max_concurrency,
                              "no_baseline": a.no_baseline})
    files = _discover(a.paths)
    if not files:
        print("no eval files found", file=sys.stderr)
        return 2
    broken = 0
    for f in files:
        try:
            _load(f)
        except Exception:  # noqa: BLE001
            broken += 1
            print(f"ERROR in {f}:\n{traceback.format_exc()}", file=sys.stderr)
    failures: list[str] = []
    mins = _parse_min(a.min)
    for r in core._registry:
        print(r.to_markdown(), end="\n\n")
        failures += r.check(mins, a.max_regression, a.max_error_rate)
    if a.json:
        Path(a.json).write_text(json.dumps([{"name": r.name, "experiment": r.experiment_id, "summary": r.summary,
                                             "comparison": r.comparison, "gate_failures": r.gate_failures}
                                            for r in core._registry], indent=2, default=str))
    for msg in failures:
        print(f"GATE FAILED: {msg}", file=sys.stderr)
    return 2 if broken else (1 if failures else 0)


def cmd_compare(a: argparse.Namespace) -> int:
    cur = load_experiment(a.store_dir, a.name, a.current)
    base = load_experiment(a.store_dir, a.name, a.baseline)
    if cur is None or base is None:
        print(f"need two experiments for {a.name!r} (found {list_experiments(Path(a.store_dir), a.name)})",
              file=sys.stderr)
        return 2
    print(json.dumps(compare_experiments(base, cur), indent=2, default=str))
    return 0


def cmd_ls(a: argparse.Namespace) -> int:
    root = Path(a.store_dir)
    names = [a.name] if a.name else sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    for n in names:
        for e in list_experiments(root, n):
            s = json.loads((root / n / e / "summary.json").read_text())
            sc = ", ".join(f"{k}={v['mean']:.3f}" for k, v in s["scores"].items() if v["mean"] is not None)
            print(f"{n}\t{e}\t{s['n_cases']} cases\t{sc}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .devserver import make_server

    srv, _ = make_server(a.host, a.port, fail_rate=a.fail_rate, fail_status=a.fail_status,
                         retry_after=a.retry_after, api_key=a.api_key, out=a.out)
    print(f"evalwright collector on http://{a.host}:{srv.server_address[1]}")
    print("  POST /v1/spans   GET /v1/spans   GET /v1/stats")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalwright", description="LLM evals and tracing")
    ap.add_argument("--version", action="version", version=f"evalwright {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run eval files")
    r.add_argument("paths", nargs="+")
    r.add_argument("--store-dir", default=".evalwright")
    r.add_argument("--max-concurrency", type=int, default=8)
    r.add_argument("--no-baseline", action="store_true")
    r.add_argument("--min", action="append", default=[], metavar="SCORE=VALUE")
    r.add_argument("--max-regression", type=float, default=None, help="fail if any score drops by more")
    r.add_argument("--max-error-rate", type=float, default=None)
    r.add_argument("--json", default=None, help="write a machine-readable report here")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare", help="compare two experiments")
    c.add_argument("name")
    c.add_argument("--baseline", default="previous")
    c.add_argument("--current", default="latest")
    c.add_argument("--store-dir", default=".evalwright")
    c.set_defaults(fn=cmd_compare)
    ls = sub.add_parser("ls", help="list experiments")
    ls.add_argument("name", nargs="?")
    ls.add_argument("--store-dir", default=".evalwright")
    ls.set_defaults(fn=cmd_ls)
    sv = sub.add_parser("serve", help="run a local collector (wire protocol v1) for development and tests")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8787)
    sv.add_argument("--fail-rate", type=float, default=0.0, help="inject failures into this share of requests")
    sv.add_argument("--fail-status", type=int, default=503)
    sv.add_argument("--retry-after", type=float, default=None)
    sv.add_argument("--api-key", default=None)
    sv.add_argument("--out", default=None, help="append accepted spans to this JSONL file")
    sv.set_defaults(fn=cmd_serve)
    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
