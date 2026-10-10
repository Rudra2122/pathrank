"""Testing framework (capture, assertions, pytest plugin) and span profiling."""
from __future__ import annotations

import asyncio
import time

import pytest

import evalwright as ew
from evalwright.scorers import ExactMatch
from evalwright.testing import assert_eval, capture_spans


@ew.traced(type="task")
def pipeline(x: int) -> int:
    return step_b(step_a(x))


@ew.traced(type="tool")
def step_a(x: int) -> int:
    return x + 1


@ew.traced(type="llm")
def step_b(x: int) -> int:
    ew.log(metrics={"tokens": 7})
    return x * 2


def test_capture_spans_tree_and_find():
    with capture_spans() as cap:
        assert pipeline(1) == 4
    assert cap.tree() == "pipeline [task]\n  step_a [tool]\n  step_b [llm]"
    assert cap.find("step_b").metrics["tokens"] == 7
    root = cap.roots()[0]
    assert [c.name for c in cap.children(root)] == ["step_a", "step_b"]
    with pytest.raises(AssertionError, match="no span named 'nope'"):
        cap.find("nope")


def test_capture_restores_previous_tracer():
    before = ew.get_tracer()
    with capture_spans():
        assert ew.get_tracer() is not before
    assert ew.get_tracer() is before


def test_ew_spans_fixture(ew_spans):
    pipeline(2)
    assert ew_spans.names().count("step_a") == 1


def test_assert_eval_passes_and_fails(tmp_path):
    data = [{"input": i, "expected": i * 2} for i in range(10)]
    good = ew.Eval("dbl", data=data, task=lambda x: x * 2, scores=[ExactMatch()], store_dir=tmp_path)
    assert assert_eval(good, min={"exact_match": 1.0}) is good
    bad = ew.Eval("dbl", data=data, task=lambda x: x * 2 if x < 5 else 0, scores=[ExactMatch()], store_dir=tmp_path)
    with pytest.raises(AssertionError, match="regressed"):
        assert_eval(bad, max_regression=0.1)


def test_pytest_plugin_end_to_end(pytester):
    """The plugin is auto-loaded via its entry point: fixture, options, terminal summary, JSON report."""
    pytester.makepyfile(test_user="""
        import evalwright as ew
        from evalwright.scorers import ExactMatch
        from evalwright.testing import assert_eval

        def test_quality():
            r = ew.Eval("plugin-eval", data=[{"input": 1, "expected": 1}], task=lambda x: x, scores=[ExactMatch()])
            assert_eval(r, min={"exact_match": 1.0})

        def test_spans(ew_spans):
            with ew.span("hello"):
                pass
            assert ew_spans.names() == ["hello"]
    """)
    store = pytester.path / "store"
    res = pytester.runpytest("-p", "no:cacheprovider", f"--ew-store-dir={store}", "--ew-report=rep.json")
    res.assert_outcomes(passed=2)
    res.stdout.fnmatch_lines(["*evalwright*", "plugin-eval: 1 cases*exact_match=1.000*"])
    assert (pytester.path / "rep.json").exists() and store.exists()


def _busy(ms: float) -> int:
    end = time.perf_counter() + ms / 1000
    n = 0
    while time.perf_counter() < end:
        n += 1
    return n


@ew.traced(profile=5)
def profiled_work() -> int:
    return _busy(20)


def test_traced_profile_attaches_report():
    with capture_spans() as cap:
        profiled_work()
    prof = cap.find("profiled_work").metadata["profile"]
    assert 0 < len(prof) <= 5
    assert any("_busy" in r["function"] for r in prof)
    assert not any("evalwright" in r["function"] and "tracer.py" in r["function"] for r in prof)
    assert all({"calls", "cumtime_ms", "tottime_ms"} <= set(r) for r in prof)


def test_profiled_block_nested_and_noop_paths():
    with capture_spans() as cap, ew.span("outer"), ew.profiled(top=3):  # noqa: SIM117 - nesting is under test
        with ew.span("inner"), ew.profiled():      # nested: covered by the outer profiler
            _busy(5)
    assert "profile" in cap.find("outer").metadata
    assert "profile" not in cap.find("inner").metadata
    with ew.profiled():                                 # outside any span: no-op
        _busy(1)


def test_profile_skipped_for_sampled_out_spans():
    with capture_spans(sample_rate=0.0) as cap:
        profiled_work()
    assert cap.spans == []


def test_async_profile():
    @ew.traced(profile=True)
    async def work() -> int:
        await asyncio.sleep(0.005)
        return _busy(5)

    with capture_spans() as cap:
        asyncio.run(work())
    assert "profile" in cap.spans[0].metadata
