from __future__ import annotations

import json
from pathlib import Path

import pytest

import evalwright as ew
from evalwright.evals import core
from evalwright.tracer import get_tracer, set_tracer

pytest_plugins = ["pytester"]

_SCHEMA = Path(__file__).resolve().parents[2] / "spec" / "span.schema.json"


@pytest.fixture
def span_schema() -> dict:
    """The shared wire schema (sdk/spec). Skips when tests run outside the repo (installed-wheel CI)."""
    if not _SCHEMA.exists():
        pytest.skip("sdk/spec not available outside the repository")
    return json.loads(_SCHEMA.read_text())


@pytest.fixture(autouse=True)
def _isolate_global_state():
    old = get_tracer()
    core._registry.clear()
    core._cli_options.clear()
    yield
    set_tracer(old)
    core._registry.clear()
    core._cli_options.clear()


@pytest.fixture
def mem():
    """A tracer exporting to memory with a fast flush; returns (tracer, exporter)."""
    exp = ew.InMemoryExporter()
    t = ew.init("test", exporter=exp, flush_interval=0.01)
    return t, exp
