"""Eval runner.

    result = Eval("qa-bot", data=[{"input": q, "expected": a}, ...], task=answer,
                  scores=[ExactMatch(), my_scorer], max_concurrency=16, trials=2)

Each (case, trial) runs concurrently (async tasks; sync tasks in worker threads) under a
semaphore, inside an "eval" span with child "task" and "score" spans, so every score links to
its full trace. Results are stored as an experiment and compared case-by-case to a baseline.
Task errors, scorer errors and timeouts are recorded per case; they never abort the run.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..serialize import to_jsonable
from ..tracer import Tracer, get_tracer
from .experiment import compare_experiments, load_experiment, new_experiment_id, resolve, save_experiment
from .stats import bootstrap_ci, mean, percentile, stderr
from .types import EvalCase, Score

Task = Callable[..., Any]
Scorer = Callable[..., Any]

_registry: list[EvalResult] = []      # collected by the CLI
_cli_options: dict[str, Any] = {}     # set by the CLI before importing eval files


@dataclass
class Hooks:
    """Second (optional) task argument: case metadata/expected and the case span."""
    metadata: dict[str, Any]
    expected: Any
    span: Any


@dataclass
class EvalResult:
    name: str
    experiment_id: str | None
    cases: list[dict[str, Any]]
    summary: dict[str, Any]
    comparison: dict[str, Any] | None = None
    path: Path | None = None
    gate_failures: list[str] = field(default_factory=list)

    def score(self, name: str) -> float | None:
        s = self.summary["scores"].get(name)
        return None if s is None else s["mean"]

    def check(self, min_scores: Mapping[str, float] | None = None, max_regression: float | None = None,
              max_error_rate: float | None = None) -> list[str]:
        """Return gate failures (empty list = pass). Used by the CLI to set the exit code."""
        fails = []
        for k, thr in (min_scores or {}).items():
            v = self.score(k)
            if v is None or v < thr:
                fails.append(f"{self.name}: {k}={v} below minimum {thr}")
        if max_regression is not None and self.comparison:
            for k, d in self.comparison["scores"].items():
                if d["diff"] is not None and d["diff"] < -max_regression:
                    fails.append(f"{self.name}: {k} regressed by {-d['diff']:.4f} (> {max_regression}) "
                                 f"vs {self.comparison['baseline']}")
        if max_error_rate is not None and self.summary["error_rate"] > max_error_rate:
            fails.append(f"{self.name}: error rate {self.summary['error_rate']:.2%} > {max_error_rate:.2%}")
        self.gate_failures = fails
        return fails

    def to_markdown(self) -> str:
        s = self.summary
        lines = [f"### {self.name} ({self.experiment_id})", "",
                 f"{s['n_cases']} cases x {s['trials']} trials, {s['errors']} errors, "
                 f"latency p50 {s['latency_ms']['p50']:.1f} ms / p95 {s['latency_ms']['p95']:.1f} ms", "",
                 "| score | mean | 95% CI | n | vs baseline |", "|---|---|---|---|---|"]
        comp = (self.comparison or {}).get("scores", {})
        for k, v in s["scores"].items():
            ci = v["ci95"]
            c = comp.get(k)
            vs = (f"{c['diff']:+.3f} ({c['improved']} up / {c['regressed']} down)" if c else "-")
            lines.append(f"| {k} | {v['mean']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {v['n']} | {vs} |"
                         if v["mean"] is not None and ci else f"| {k} | - | - | 0 | {vs} |")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()


def _as_case(x: Any) -> EvalCase:
    if isinstance(x, EvalCase):
        return x
    if isinstance(x, Mapping) and "input" in x:
        return EvalCase(x["input"], x.get("expected"), dict(x.get("metadata") or {}), x.get("id"))
    return EvalCase(x)


def _scorer_name(s: Scorer) -> str:
    return str(getattr(s, "name", None) or getattr(s, "__name__", None) or type(s).__name__)


def _call_kwargs(fn: Callable[..., Any], available: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters      # works for functions and callable instances
    except (TypeError, ValueError):
        return available
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return available
    return {k: v for k, v in available.items() if k in params}


def _normalize(name: str, r: Any) -> list[Score]:
    if r is None:
        return [Score(name, None)]
    if isinstance(r, Score):
        return [r]
    if isinstance(r, bool):
        return [Score(name, float(r))]
    if isinstance(r, (int, float)):
        return [Score(name, float(r))]
    if isinstance(r, Mapping):
        return [Score(str(k), None if v is None else float(v)) for k, v in r.items()]
    if isinstance(r, Sequence) and all(isinstance(x, Score) for x in r):
        return list(r)
    raise TypeError(f"scorer {name} returned unsupported {type(r).__name__}")


def _is_async(fn: Callable[..., Any]) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    call = getattr(type(fn), "__call__", None)  # noqa: B004 - need the method object to see `async def`
    return not inspect.isfunction(fn) and call is not None and inspect.iscoroutinefunction(call)


async def _maybe_await(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    if _is_async(fn):
        return await fn(**kwargs)
    out = await asyncio.to_thread(fn, **kwargs)     # copies contextvars -> spans nest correctly
    if inspect.isawaitable(out):
        out = await out
    return out


async def eval_async(name: str, data: Iterable[Any] | Callable[[], Iterable[Any]], task: Task,
                     scores: Sequence[Scorer] = (), *, max_concurrency: int = 8, trials: int = 1,
                     timeout: float | None = None, tracer: Tracer | None = None,
                     store_dir: str | Path | None = ".evalwright", baseline: str | None = "latest",
                     metadata: Mapping[str, Any] | None = None, seed: int = 0) -> EvalResult:
    tracer = tracer or get_tracer()
    max_concurrency = int(_cli_options.get("max_concurrency", max_concurrency))
    rows = data() if callable(data) else data
    cases = [_as_case(x) for x in rows]
    if not cases:
        raise ValueError(f"eval {name!r} has no cases")
    n_params = len([p for p in inspect.signature(task).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    sem = asyncio.Semaphore(max_concurrency)
    t_start = time.perf_counter()

    async def run_one(case: EvalCase, trial: int) -> dict[str, Any]:
        async with sem:
            rec: dict[str, Any] = {"case_id": case.case_id(), "trial": trial, "input": to_jsonable(case.input),
                                   "expected": to_jsonable(case.expected), "metadata": to_jsonable(case.metadata),
                                   "output": None, "scores": {}, "score_metadata": {}, "error": None}
            t0 = time.perf_counter()
            async with tracer.span(name, "eval", input=case.input, expected=case.expected,
                                   metadata={**case.metadata, "trial": trial, "case_id": rec["case_id"]}) as root:
                rec["trace_id"] = root.trace_id
                try:
                    async with tracer.span("task", "task") as ts:
                        args = ((case.input,) if n_params < 2
                                else (case.input, Hooks(case.metadata, case.expected, root)))
                        coro = task(*args) if _is_async(task) else asyncio.to_thread(task, *args)
                        output = await (asyncio.wait_for(coro, timeout) if timeout else coro)
                        ts.log(output=output)
                    rec["output"] = to_jsonable(output)
                except asyncio.TimeoutError:
                    rec["error"] = {"type": "Timeout", "message": f"task exceeded {timeout}s"}
                    root.status = "error"
                    output = None
                except Exception as e:  # noqa: BLE001 - recorded, run continues
                    rec["error"] = {"type": type(e).__name__, "message": str(e)[:1000]}
                    output = None
                rec["duration_ms"] = (time.perf_counter() - t0) * 1000
                if rec["error"] is None:
                    avail = {"input": case.input, "output": output, "expected": case.expected,
                             "metadata": case.metadata}
                    for sc in scores:
                        sname = _scorer_name(sc)
                        try:
                            async with tracer.span(sname, "score") as ss:
                                res = _normalize(sname, await _maybe_await(sc, _call_kwargs(sc, avail)))
                                ss.log(output=[{"name": x.name, "score": x.score} for x in res])
                            for x in res:
                                rec["scores"][x.name] = x.score
                                if x.metadata:
                                    rec["score_metadata"][x.name] = to_jsonable(x.metadata)
                        except Exception as e:  # noqa: BLE001
                            rec["scores"][sname] = None
                            rec.setdefault("scorer_errors", {})[sname] = f"{type(e).__name__}: {str(e)[:300]}"
                root.log(output=output, scores=rec["scores"])
            return rec

    results = await asyncio.gather(*(run_one(c, t) for c in cases for t in range(trials)))
    results.sort(key=lambda r: (r["case_id"], r["trial"]))
    wall = time.perf_counter() - t_start
    summary = summarize(results, trials, wall, seed)
    summary["metadata"] = to_jsonable(dict(metadata or {}))

    exp_id = comparison = path = None
    store = _cli_options.get("store_dir", store_dir)
    if _cli_options.get("no_baseline"):
        baseline = None
    if store is not None:
        store = Path(store)
        base = None
        if baseline is not None:
            ref = resolve(store, name, baseline)
            base = load_experiment(store, name, ref) if ref else None
        exp_id = new_experiment_id()
        path = save_experiment(store, name, exp_id, results, summary)
        if base is not None:
            comparison = compare_experiments(base, {"id": exp_id, "cases": results})
    tracer.flush(timeout=2.0)
    res = EvalResult(name, exp_id, results, summary, comparison, path)
    _registry.append(res)
    return res


def summarize(results: list[dict[str, Any]], trials: int, wall_s: float, seed: int = 0) -> dict[str, Any]:
    names = sorted({k for r in results for k in r["scores"]})
    scores = {}
    for k in names:
        vals = [r["scores"][k] for r in results if r["scores"].get(k) is not None]
        scores[k] = {"mean": mean(vals), "stderr": stderr(vals), "ci95": bootstrap_ci(vals, seed=seed),
                     "n": len(vals), "min": min(vals) if vals else None, "max": max(vals) if vals else None}
    lat = [r["duration_ms"] for r in results]
    errors = sum(r["error"] is not None for r in results)
    return {"n_cases": len({r["case_id"] for r in results}), "trials": trials, "n_runs": len(results),
            "errors": errors, "error_rate": errors / len(results),
            "scorer_errors": sum(len(r.get("scorer_errors", {})) for r in results),
            "scores": scores, "latency_ms": {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95)},
            "wall_seconds": wall_s}


def Eval(name: str, data: Iterable[Any] | Callable[[], Iterable[Any]], task: Task,  # noqa: N802 - public API
         scores: Sequence[Scorer] = (), **kw: Any) -> EvalResult:
    """Synchronous entry point. Inside a running event loop use ``await eval_async(...)``."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(eval_async(name, data, task, scores, **kw))
    raise RuntimeError("Eval() was called inside a running event loop; use `await evalwright.eval_async(...)`")
