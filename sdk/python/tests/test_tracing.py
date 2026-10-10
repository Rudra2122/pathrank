from __future__ import annotations

import asyncio
import threading

import pytest

import evalwright as ew
from evalwright.tracer import Tracer


def by_name(exp):
    return {s["name"]: s for s in exp.spans}


def test_nesting_metrics_and_timing(mem):
    t, exp = mem
    with t.span("root", type="task", input={"q": 1}) as r:
        with t.span("child", type="llm") as c:
            c.log(output="x", metrics={"tokens": 5}, tags=["a", "a", "b"])
        r.log(metadata={"k": "v"})
    t.flush()
    s = by_name(exp)
    assert s["child"]["parent_id"] == s["root"]["span_id"]
    assert s["child"]["trace_id"] == s["root"]["trace_id"]
    assert s["child"]["metrics"] == {"tokens": 5.0} and s["child"]["tags"] == ["a", "b"]
    assert s["root"]["metadata"] == {"k": "v"} and s["root"]["duration_ms"] >= s["child"]["duration_ms"]
    assert s["root"]["cpu_ms"] is not None and s["root"]["project"] == "test"


def test_errors_recorded_and_reraised(mem):
    t, exp = mem
    with pytest.raises(ZeroDivisionError), t.span("bad"):
        _ = 1 / 0
    t.flush()
    s = by_name(exp)["bad"]
    assert s["status"] == "error" and s["error"]["type"] == "ZeroDivisionError" and "1 / 0" in s["error"]["stack"]


def test_decorator_sync_async_and_metadata(mem):
    t, exp = mem

    @t.traced(type="llm", name="f")
    def f(x, y=2):
        """doc"""
        return x + y

    @t.traced(name="g_async")
    async def g(x):
        return f(x)

    assert f.__name__ == "f" and f.__doc__ == "doc"
    assert asyncio.run(g(1)) == 3
    t.flush()
    s = by_name(exp)
    assert s["f"]["parent_id"] == s["g_async"]["span_id"]
    assert s["f"]["input"] == {"x": 1} and s["f"]["output"] == 3 and s["f"]["type"] == "llm"


def test_capture_flags(mem):
    t, exp = mem

    @t.traced(capture_args=False, capture_output=False)
    def secret(pw):
        return pw

    secret("hunter2")
    t.flush()
    assert by_name(exp)["test_capture_flags.<locals>.secret"]["input"] is None


def test_async_tasks_are_isolated(mem):
    t, exp = mem

    async def worker(i):
        async with t.span(f"w{i}"):
            await asyncio.sleep(0.01)
            async with t.span(f"w{i}.inner"):
                await asyncio.sleep(0.01)

    async def main():
        await asyncio.gather(*(worker(i) for i in range(10)))

    asyncio.run(main())
    t.flush()
    s = by_name(exp)
    for i in range(10):
        assert s[f"w{i}.inner"]["parent_id"] == s[f"w{i}"]["span_id"]
        assert s[f"w{i}"]["parent_id"] is None


def test_threads_need_explicit_context(mem):
    t, _ = mem
    out = {}
    with t.span("root") as r:
        th1 = threading.Thread(target=lambda: out.setdefault("plain", ew.current_span()))
        th2 = threading.Thread(target=ew.run_in_thread_context(lambda: out.setdefault("ctx", ew.current_span())))
        th1.start(), th2.start(), th1.join(), th2.join()
    assert out["plain"] is None and out["ctx"] is r


def test_sync_generator_does_not_leak_context(mem):
    t, exp = mem

    @t.traced
    def gen():
        with t.span("inside_gen"):
            pass
        yield 1
        yield 2

    with t.span("consumer"):
        g = gen()
        next(g)
        with t.span("between_yields"):      # generator is suspended here
            pass
        assert list(g) == [2]
    t.flush()
    s = by_name(exp)
    gname = "test_sync_generator_does_not_leak_context.<locals>.gen"
    assert s["between_yields"]["parent_id"] == s["consumer"]["span_id"]
    assert s["inside_gen"]["parent_id"] == s[gname]["span_id"]
    assert s[gname]["metrics"]["items"] == 2


def test_async_generator_does_not_leak_context(mem):
    t, exp = mem

    @t.traced
    async def agen():
        for i in range(3):
            yield i

    async def main():
        async with t.span("consumer"):
            it = agen().__aiter__()
            await it.__anext__()
            async with t.span("between"):
                pass
            return [x async for x in it]

    assert asyncio.run(main()) == [1, 2]
    t.flush()
    s = by_name(exp)
    assert s["between"]["parent_id"] == s["consumer"]["span_id"]


def test_generator_error_recorded(mem):
    t, exp = mem

    @t.traced
    def bad():
        yield 1
        raise KeyError("k")

    with pytest.raises(KeyError):
        list(bad())
    t.flush()
    assert next(s for s in exp.spans if s["name"].endswith("bad"))["status"] == "error"


def test_sampling_is_per_trace_and_deterministic():
    exp = ew.InMemoryExporter()
    t = ew.init("s", exporter=exp, sample_rate=0.3, flush_interval=0.01)
    for _ in range(400):
        with t.span("root"), t.span("child"):
            pass
    t.flush()
    roots = [s for s in exp.spans if s["name"] == "root"]
    kids = [s for s in exp.spans if s["name"] == "child"]
    assert 60 < len(roots) < 180                              # ~120 expected
    assert {k["trace_id"] for k in kids} == {r["trace_id"] for r in roots}   # whole traces kept or dropped
    assert t._sampled("ab" * 16) == t._sampled("ab" * 16)
    with pytest.raises(ValueError):
        Tracer(sample_rate=1.5)


def test_disabled_and_uninitialized_are_noops():
    ew.set_tracer(Tracer(enabled=False))
    with ew.span("x") as s:
        s.log(output=1)
        ew.log(metadata={"a": 1})
    assert ew.flush() is True
    t = ew.init("d", exporter=ew.InMemoryExporter(), enabled=False)
    with t.span("y"):
        pass
    assert t.stats()["spans_started"] == 1


def test_module_level_traced_binds_lazily():
    @ew.traced(name="late")
    def f():
        return 7

    exp = ew.InMemoryExporter()
    t = ew.init("lazy", exporter=exp, flush_interval=0.01)     # init AFTER decoration
    assert f() == 7
    with ew.span("outer"):
        ew.log(metadata={"via": "log"})
    t.flush()
    names = {s["name"]: s for s in exp.spans}
    assert "late" in names and names["outer"]["metadata"] == {"via": "log"}


def test_end_is_idempotent_and_start_span(mem):
    t, exp = mem
    s = t.start_span("manual", "tool")
    s.end()
    s.end()
    t.flush()
    assert [x["name"] for x in exp.spans].count("manual") == 1 and s.ended
    assert "manual" in repr(s)


def test_env_exporter_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALWRIGHT_EXPORT", f"jsonl:{tmp_path / 't.jsonl'}")
    t = ew.init("envtest", flush_interval=0.01)
    with t.span("a"):
        pass
    t.flush()
    assert (tmp_path / "t.jsonl").read_text().count('"name":"a"') == 1
    for spec, cls in (("console", ew.ConsoleExporter), ("memory", ew.InMemoryExporter),
                      ("otlp:http://x:4318/v1/traces", ew.OtlpHttpExporter), ("http:http://x/ingest", ew.HttpExporter)):
        monkeypatch.setenv("EVALWRIGHT_EXPORT", spec)
        assert isinstance(ew.init("e").processor.exporter, cls)
    monkeypatch.setenv("EVALWRIGHT_EXPORT", "http:")
    with pytest.raises(ValueError):
        ew.init("e")
    monkeypatch.setenv("EVALWRIGHT_DISABLED", "1")
    monkeypatch.setenv("EVALWRIGHT_EXPORT", "memory")
    assert ew.init("e").processor is None


def test_serializer_failure_never_reaches_user(mem):
    t, exp = mem

    class Exploding:
        def __getattribute__(self, name):
            raise RuntimeError("no")

    with t.span("x") as s:
        s.log(output=Exploding())
    t.flush()
    assert by_name(exp)["x"]["output"] is not None


def test_method_binding_drops_self_and_handles_varargs(mem):
    t, exp = mem

    class Svc:
        @t.traced(name="m")
        def m(self, a, b=1):
            return a + b

        @t.traced(name="v")
        def v(self, *xs, **kw):
            return sum(xs)

    assert Svc().m(1, b=2) == 3 and Svc().v(1, 2, k=3) == 3
    t.flush()
    s = by_name(exp)
    assert s["m"]["input"] == {"a": 1, "b": 2}
    assert s["v"]["input"] == {"xs": [1, 2], "kw": {"k": 3}}
