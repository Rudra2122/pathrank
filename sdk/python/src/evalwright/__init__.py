"""evalwright: tracing and evaluation for LLM applications.

    import evalwright as ew
    ew.init("my-app")                                  # JSONL by default; OTLP / HTTP / custom exporters

    @ew.traced(type="llm")
    def answer(question): ...

    with ew.span("retrieve", type="retrieval", input=query) as s:
        s.log(output=docs, metrics={"n_docs": len(docs)})

    ew.Eval("qa", data=cases, task=answer, scores=[ew.scorers.ExactMatch()])
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import scorers
from ._version import __version__
from .evals import Eval, EvalCase, EvalResult, Score, compare_experiments, eval_async, load_experiment
from .export import (
    BatchProcessor,
    CircuitBreaker,
    ConsoleExporter,
    Exporter,
    ExportResult,
    HttpExporter,
    InMemoryExporter,
    JsonlExporter,
    OtlpHttpExporter,
    RetryPolicy,
)
from .profiling import profiled
from .serialize import Limits, default_redactor, to_jsonable
from .span import Span, SpanType
from .tracer import Tracer, current_span, get_tracer, run_in_thread_context, set_tracer, traced


def _exporter_from_env(project: str) -> Exporter:
    spec = os.environ.get("EVALWRIGHT_EXPORT", "")
    kind, _, arg = spec.partition(":")
    if kind == "otlp":
        return OtlpHttpExporter(arg or "http://localhost:4318/v1/traces", service_name=project)
    if kind == "http":
        if not arg:
            raise ValueError("EVALWRIGHT_EXPORT=http:<url> needs a URL")
        return HttpExporter(arg, api_key=os.environ.get("EVALWRIGHT_API_KEY"))
    if kind == "console":
        return ConsoleExporter()
    if kind == "memory":
        return InMemoryExporter()
    return JsonlExporter(arg or Path(".evalwright") / "traces" / f"{project}.jsonl")


def init(project: str = "default", *, exporter: Exporter | None = None, sample_rate: float | None = None,
         redact_emails: bool = False, limits: Limits | None = None, enabled: bool | None = None,
         **processor_kw: Any) -> Tracer:
    """Install the process-global tracer. Safe to call again (the old tracer is flushed).

    Environment: EVALWRIGHT_EXPORT=jsonl:<path>|otlp:<url>|http:<url>|console|memory,
    EVALWRIGHT_API_KEY, EVALWRIGHT_SAMPLE_RATE, EVALWRIGHT_DISABLED=1.
    """
    if enabled is None:
        enabled = os.environ.get("EVALWRIGHT_DISABLED", "").lower() not in ("1", "true", "yes")
    if sample_rate is None:
        sample_rate = float(os.environ.get("EVALWRIGHT_SAMPLE_RATE", "1.0"))
    exp = exporter if exporter is not None else _exporter_from_env(project)
    t = Tracer(project, BatchProcessor(exp, **processor_kw) if enabled else None, sample_rate=sample_rate,
               redactor=default_redactor(redact_emails), limits=limits or Limits(), enabled=enabled)
    return set_tracer(t)


def span(name: str, type: SpanType = "function", **fields: Any) -> Any:
    """Open a span on the global tracer: ``with evalwright.span("step") as s: ...``"""
    return get_tracer().span(name, type, **fields)


def log(**fields: Any) -> None:
    """Log onto the current span (no-op outside a span)."""
    s = current_span()
    if s is not None:
        s.log(**fields)


def flush(timeout: float | None = 5.0) -> bool:
    return get_tracer().flush(timeout)


__all__ = ["BatchProcessor", "CircuitBreaker", "ConsoleExporter", "Eval", "EvalCase", "EvalResult", "ExportResult",
           "Exporter", "HttpExporter", "InMemoryExporter", "JsonlExporter", "Limits", "OtlpHttpExporter",
           "RetryPolicy", "Score", "Span", "Tracer", "__version__", "compare_experiments", "current_span",
           "default_redactor", "eval_async", "flush", "get_tracer", "init", "load_experiment", "log", "profiled",
           "run_in_thread_context", "scorers", "set_tracer", "span", "to_jsonable", "traced"]
