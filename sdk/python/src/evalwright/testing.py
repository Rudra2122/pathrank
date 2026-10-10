"""Test helpers: capture spans in-process and assert on traces and eval quality.

    from evalwright.testing import capture_spans, assert_eval

    def test_agent_traces():
        with capture_spans() as cap:
            run_agent("hi")
        assert cap.tree() == "agent [task]\\n  draft [llm]\\n  guardrails [function]"
        assert cap.find("draft").metrics["tokens"] > 0

    def test_agent_quality():
        assert_eval(Eval("agent", data, task, scores), min={"judge": 0.8}, max_regression=0.02)

With pytest, the `ew_spans` fixture is the same capture (plugin: pytest_evalwright).
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from .evals.core import EvalResult
from .export.exporters import InMemoryExporter
from .tracer import Tracer, get_tracer, set_tracer


class SpanCapture:
    """Spans recorded while active, as attribute-access records (`s.name`, `s.metrics`, ...)."""

    def __init__(self, tracer: Tracer, exporter: InMemoryExporter) -> None:
        self.tracer, self._exporter = tracer, exporter

    @property
    def spans(self) -> list[Any]:
        self.tracer.flush(5.0)
        return [SimpleNamespace(**r) for r in self._exporter.spans]

    def names(self) -> list[str]:
        return [s.name for s in self.spans]

    def named(self, name: str) -> list[Any]:
        return [s for s in self.spans if s.name == name]

    def find(self, name: str) -> Any:
        found = self.named(name)
        if not found:
            raise AssertionError(f"no span named {name!r}; recorded: {sorted(set(self.names()))}")
        return found[0]

    def children(self, span: Any) -> list[Any]:
        return sorted((s for s in self.spans if s.parent_id == span.span_id), key=lambda s: s.start_ns)

    def roots(self) -> list[Any]:
        spans = self.spans
        ids = {s.span_id for s in spans}
        return sorted((s for s in spans if s.parent_id is None or s.parent_id not in ids), key=lambda s: s.start_ns)

    def tree(self, *, show_status: bool = False) -> str:
        """Indented `name [type]` per span, children in start order. Stable enough to compare in tests."""
        spans = self.spans
        by_parent: dict[str | None, list[Any]] = {}
        ids = {s.span_id for s in spans}
        for s in spans:
            by_parent.setdefault(s.parent_id if s.parent_id in ids else None, []).append(s)
        lines: list[str] = []

        def walk(parent: str | None, depth: int) -> None:
            for s in sorted(by_parent.get(parent, []), key=lambda x: x.start_ns):
                status = f" {s.status}" if show_status else ""
                lines.append(f"{'  ' * depth}{s.name} [{s.type}]{status}")
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)

    def clear(self) -> None:
        self.tracer.flush(5.0)
        self._exporter.spans.clear()


@contextmanager
def capture_spans(project: str = "test", **init_kw: Any) -> Iterator[SpanCapture]:
    """Route all tracing to memory for the duration; the previous tracer is restored after."""
    from . import init  # package-level init (late import: evalwright/__init__ imports more than we need)

    old = get_tracer()
    exp = InMemoryExporter()
    t = init(project, exporter=exp, flush_interval=0.01, **init_kw)
    try:
        yield SpanCapture(t, exp)
    finally:
        t.flush(5.0)
        set_tracer(old)


def assert_eval(result: EvalResult, *, min: Mapping[str, float] | None = None,
                max_regression: float | None = None, max_error_rate: float | None = None) -> EvalResult:
    """Raise AssertionError (with the summary) if any gate fails; return the result otherwise."""
    fails = result.check(min, max_regression, max_error_rate)
    if fails:
        raise AssertionError("eval gates failed:\n  " + "\n  ".join(fails) + f"\n\n{result}")
    return result
