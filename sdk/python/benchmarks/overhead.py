"""Tracing overhead profiler.

Measures, per call of a tiny function:
  baseline            undecorated call
  disabled            @traced with tracing disabled (what users pay before init())
  sampled_out         @traced on a tracer with sample_rate=0
  traced_small        @traced, exported to memory, small args/result
  traced_large        @traced, 20 KB string + nested dict (serialization + truncation path)
  nested_3            three nested spans per call
Reports median ns/call and overhead vs baseline, bytes allocated per span (tracemalloc), and a
cProfile hotspot table. `--check budget.json` exits 1 if any overhead exceeds its budget (CI guard).

  python benchmarks/overhead.py --json out.json --profile --check benchmarks/budget.json
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import evalwright as ew
from evalwright.export import BatchProcessor, ExportResult
from evalwright.tracer import Tracer, set_tracer


class NullExporter:
    def export(self, batch: list[dict[str, Any]]) -> ExportResult:
        return ExportResult.ok()

    def close(self) -> None:
        pass


def bench(fn: Callable[[], Any], n: int, repeats: int = 7) -> float:
    for _ in range(min(n, 2000)):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        samples.append((time.perf_counter_ns() - t0) / n)
    return statistics.median(samples)


def make_cases() -> dict[str, tuple[Tracer, Callable[[], Any]]]:
    big = {"text": "x" * 20_000, "rows": [{"i": i, "v": [i] * 10} for i in range(100)]}

    def add(a: int, b: int) -> int:
        return a + b

    def echo(payload: dict[str, Any]) -> int:
        return len(payload)

    off = Tracer(enabled=False)
    out = Tracer("bench", BatchProcessor(NullExporter(), max_queue=1_000_000, flush_interval=0.05), sample_rate=0.0)
    on = Tracer("bench", BatchProcessor(NullExporter(), max_queue=1_000_000, flush_interval=0.05))
    f_off, f_out, f_on, f_big = off.traced(add), out.traced(add), on.traced(add), on.traced(echo)

    def nested() -> None:
        with on.span("a"), on.span("b"), on.span("c"):
            pass

    return {"baseline": (off, lambda: add(1, 2)), "disabled": (off, lambda: f_off(1, 2)),
            "sampled_out": (out, lambda: f_out(1, 2)), "traced_small": (on, lambda: f_on(1, 2)),
            "traced_large": (on, lambda: f_big(big)), "nested_3": (on, nested)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", default=None, help="budget JSON: {case: max_overhead_us}")
    a = ap.parse_args()
    cases = make_cases()
    res: dict[str, Any] = {"python": sys.version.split()[0], "evalwright": ew.__version__, "n": a.n, "cases": {}}
    base = None
    for name, (tracer, fn) in cases.items():
        n = a.n if name != "traced_large" else max(500, a.n // 20)
        ns = bench(fn, n)
        tracer.flush(10)
        if name == "baseline":
            base = ns
        res["cases"][name] = {"ns_per_call": round(ns), "overhead_us": round((ns - (base or 0)) / 1000, 3)}
    # memory per span on the exporting path
    tracer, fn = cases["traced_small"]
    tracer.flush(10)
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for _ in range(5000):
        fn()
    snap1 = tracemalloc.take_snapshot()
    tracer.flush(10)
    tracemalloc.stop()
    alloc = sum(s.size_diff for s in snap1.compare_to(snap0, "filename") if s.size_diff > 0)
    res["bytes_retained_per_span_before_flush"] = round(alloc / 5000)
    print(f"{'case':14s} {'ns/call':>10s} {'overhead us':>12s}")
    for k, v in res["cases"].items():
        print(f"{k:14s} {v['ns_per_call']:>10,} {v['overhead_us']:>12.3f}")
    print(f"bytes held per queued span: {res['bytes_retained_per_span_before_flush']}")
    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(20_000):
            fn()
        pr.disable()
        tracer.flush(10)
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(12)
        res["profile_top"] = buf.getvalue()
        print(buf.getvalue())
    for t, _ in cases.values():
        t.shutdown(2)
    set_tracer(Tracer(enabled=False))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
    if a.check:
        with open(a.check) as f:
            budget = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        over = {k: (res["cases"][k]["overhead_us"], lim) for k, lim in budget.items()
                if res["cases"][k]["overhead_us"] > lim}
        if over:
            print(f"OVER BUDGET: {over}", file=sys.stderr)
            return 1
        print("within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
