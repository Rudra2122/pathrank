"""pytest plugin for evalwright, auto-registered via the `pytest11` entry point.

Kept as a separate top-level module that imports evalwright lazily: merely having the SDK
installed adds nothing to a pytest session's startup, and coverage of evalwright itself is
measured correctly.

Fixtures
  ew_spans              SpanCapture for the test (see evalwright.testing)

Options
  --ew-store-dir DIR    store eval experiments run in tests here
  --ew-no-baseline      don't compare evals with stored baselines
  --ew-report FILE      write every eval run during the session (summary + gate failures) as JSON

Evals run by tests are summarized at the end of the terminal report.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from evalwright.evals.core import EvalResult
    from evalwright.testing import SpanCapture

_START = "_evalwright_registry_start"


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("evalwright")
    g.addoption("--ew-store-dir", default=None, help="directory for eval experiments run in tests")
    g.addoption("--ew-no-baseline", action="store_true", help="don't compare evals with stored baselines")
    g.addoption("--ew-report", default=None, help="write a JSON report of evals run during the session")


def pytest_configure(config: pytest.Config) -> None:
    core = sys.modules.get("evalwright.evals.core")     # only if already imported; never import eagerly
    config.stash[_start_key] = len(core._registry) if core is not None else 0
    store, no_base = config.getoption("--ew-store-dir", None), config.getoption("--ew-no-baseline", False)
    if store or no_base:
        from evalwright.evals import core as c  # user asked for eval options: import is expected
        if store:
            c._cli_options["store_dir"] = store
        if no_base:
            c._cli_options["no_baseline"] = True


_start_key = pytest.StashKey[int]()


@pytest.fixture
def ew_spans() -> Iterator[SpanCapture]:
    """Capture every span emitted during the test."""
    from evalwright.testing import capture_spans

    with capture_spans("pytest") as cap:
        yield cap


def _session_results(config: pytest.Config) -> list[EvalResult]:
    core = sys.modules.get("evalwright.evals.core")
    if core is None:
        return []
    results: list[EvalResult] = core._registry[config.stash.get(_start_key, 0):]
    return results


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    results = _session_results(config)
    if not results:
        return
    tr = terminalreporter
    tr.section("evalwright")
    for r in results:
        scores = ", ".join(f"{k}={v['mean']:.3f}" for k, v in r.summary["scores"].items() if v["mean"] is not None)
        flag = " FAILED GATES" if r.gate_failures else ""
        tr.write_line(f"{r.name}: {r.summary['n_cases']} cases, errors {r.summary['error_rate']:.1%}, {scores}{flag}")
    path = config.getoption("--ew-report", None)
    if path:
        with open(path, "w") as f:
            json.dump([{"name": r.name, "experiment_id": r.experiment_id, "summary": r.summary,
                        "comparison": r.comparison, "gate_failures": r.gate_failures} for r in results],
                      f, indent=2, default=str)
        tr.write_line(f"report written to {path}")
