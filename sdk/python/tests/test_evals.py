from __future__ import annotations

import asyncio
import threading
import time

import pytest

import evalwright as ew
from evalwright.evals import EvalCase, Hooks, compare_experiments, load_experiment
from evalwright.scorers import Contains, ExactMatch, JSONValid, Levenshtein, LLMJudge, NumericCloseness, RegexMatch

DATA = [{"input": "a", "expected": "A"}, {"input": "b", "expected": "B"}, {"input": "c", "expected": "C"}]


def test_basic_eval_summary_and_storage(tmp_path, mem):
    t, exp = mem
    r = ew.Eval("upper", DATA, str.upper, [ExactMatch(), Levenshtein()], store_dir=tmp_path)
    assert r.score("exact_match") == 1.0 and r.summary["n_cases"] == 3 and r.summary["errors"] == 0
    assert r.summary["scores"]["exact_match"]["ci95"] == (1.0, 1.0)
    assert (tmp_path / "upper" / r.experiment_id / "cases.jsonl").exists()
    assert load_experiment(tmp_path, "upper")["id"] == r.experiment_id
    assert "| exact_match | 1.000" in str(r)
    t.flush()
    types = {s["type"] for s in exp.spans}
    assert {"eval", "task", "score"} <= types
    root = next(s for s in exp.spans if s["type"] == "eval")
    assert root["scores"] == {"exact_match": 1.0, "levenshtein": 1.0}
    assert all(s["trace_id"] == root["trace_id"] for s in exp.spans if s["trace_id"] == root["trace_id"])


def test_async_task_trials_and_concurrency_limit(tmp_path):
    live, peak = 0, 0
    lock = threading.Lock()

    async def task(x):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        await asyncio.sleep(0.02)
        with lock:
            live -= 1
        return x.upper()

    r = ew.Eval("conc", [{"input": str(i), "expected": str(i)} for i in range(20)], task, [ExactMatch()],
                max_concurrency=4, trials=2, store_dir=tmp_path)
    assert peak <= 4 and r.summary["n_runs"] == 40 and r.summary["trials"] == 2


def test_sync_tasks_run_concurrently_in_threads(tmp_path):
    def slow(x):
        time.sleep(0.05)
        return x

    t0 = time.perf_counter()
    ew.Eval("threads", [{"input": i} for i in range(16)], slow, [], max_concurrency=16, store_dir=tmp_path)
    assert time.perf_counter() - t0 < 0.6          # serial would be 0.8s


def test_errors_timeouts_and_scorer_failures_are_recorded(tmp_path):
    async def task(x):
        if x == "boom":
            raise ValueError("task failed")
        if x == "slow":
            await asyncio.sleep(1)
        return x

    def bad_scorer(output):
        raise RuntimeError("scorer bug")

    def out_of_range(output):
        return 3.0

    r = ew.Eval("errs", [{"input": "ok", "expected": "ok"}, {"input": "boom"}, {"input": "slow"}], task,
                [ExactMatch(), bad_scorer, out_of_range], timeout=0.1, store_dir=tmp_path)
    by = {c["input"]: c for c in r.cases}
    assert by["boom"]["error"]["type"] == "ValueError" and by["slow"]["error"]["type"] == "Timeout"
    assert by["ok"]["scores"]["exact_match"] == 1.0 and by["ok"]["scores"]["bad_scorer"] is None
    assert "outside [0, 1]" in by["ok"]["scorer_errors"]["out_of_range"]
    assert r.summary["errors"] == 2 and r.summary["scorer_errors"] == 2
    assert r.check(max_error_rate=0.5) and r.check(max_error_rate=0.7) == []


def test_scorer_signatures_and_return_types(tmp_path):
    def only_output(output):
        return output == "A"

    def with_meta(input, output, expected, metadata):
        return {"len_ok": 1.0 if len(output) == 1 else 0.0, "tagged": float(metadata.get("tag") == "x")}

    def many(output, **kw):
        return [ew.Score("s1", 1.0, {"why": "ok"}), ew.Score("s2", None)]

    async def async_scorer(output):
        return 0.5

    r = ew.Eval("sig", [EvalCase("a", "A", {"tag": "x"}, id="case-a")], str.upper,
                [only_output, with_meta, many, async_scorer], store_dir=tmp_path)
    c = r.cases[0]
    assert c["case_id"] == "case-a"
    assert c["scores"] == {"only_output": 1.0, "len_ok": 1.0, "tagged": 1.0, "s1": 1.0, "s2": None,
                           "async_scorer": 0.5}
    assert c["score_metadata"]["s1"] == {"why": "ok"}
    assert r.summary["scores"]["s2"]["n"] == 0


def test_hooks_argument_and_callable_data(tmp_path):
    def task(x, hooks: Hooks):
        hooks.span.log(metadata={"seen": True})
        return f"{x}:{hooks.metadata['k']}:{hooks.expected}"

    r = ew.Eval("hooks", lambda: [{"input": "i", "expected": "e", "metadata": {"k": "v"}}], task,
                [ExactMatch()], store_dir=tmp_path)
    assert r.cases[0]["output"] == "i:v:e"


def test_baseline_comparison_and_regression_gate(tmp_path):
    ew.Eval("cmp", DATA, str.upper, [ExactMatch()], store_dir=tmp_path)
    worse = ew.Eval("cmp", DATA, lambda x: "A", [ExactMatch()], store_dir=tmp_path)
    c = worse.comparison["scores"]["exact_match"]
    assert c["regressed"] == 2 and c["improved"] == 0 and c["unchanged"] == 1
    assert round(c["diff"], 4) == round(-2 / 3, 4)
    assert worse.check(max_regression=0.7) == [] and worse.check(max_regression=0.1)
    assert worse.check(min_scores={"exact_match": 0.5}) and worse.check(min_scores={"missing": 0.1})
    assert worse.check(min_scores={"exact_match": 0.3}) == []
    base = load_experiment(tmp_path, "cmp", "previous")
    cur = load_experiment(tmp_path, "cmp", "latest")
    assert compare_experiments(base, cur)["common_cases"] == 3
    no_base = ew.Eval("cmp", DATA, str.upper, [ExactMatch()], store_dir=tmp_path, baseline=None)
    assert no_base.comparison is None
    nostore = ew.Eval("cmp2", DATA, str.upper, [ExactMatch()], store_dir=None)
    assert nostore.experiment_id is None


def test_case_ids_stable_and_validation():
    assert EvalCase({"q": 1}, "a").case_id() == EvalCase({"q": 1}, "a").case_id()
    assert EvalCase("x").case_id() != EvalCase("y").case_id()
    with pytest.raises(ValueError):
        ew.Eval("empty", [], str.upper, store_dir=None)


def test_eval_inside_running_loop_raises_helpfully():
    async def main():
        with pytest.raises(RuntimeError, match="eval_async"):
            ew.Eval("x", DATA, str.upper, store_dir=None)
        r = await ew.eval_async("x", DATA, str.upper, [ExactMatch()], store_dir=None)
        return r.score("exact_match")

    assert asyncio.run(main()) == 1.0


def test_builtin_scorers():
    assert ExactMatch(case_sensitive=False)("Hi ", "hi") == 1.0 and ExactMatch()(1, None) is None
    assert Contains()("Hello World", ["hello", "world"]) == 1.0
    assert Contains(mode="any")("abc", ["x", "b"]) == 1.0
    assert Contains(mode="fraction")("abc", ["a", "x"]) == 0.5 and Contains()("x", None) is None
    with pytest.raises(ValueError):
        Contains(mode="bad")
    assert JSONValid(("a",))('{"a": 1}') == 1.0 and JSONValid()("not json") == 0.0
    assert JSONValid()({"a": 1}) == 1.0 and JSONValid(("b",))({"a": 1}) == 0.0
    assert Levenshtein()("kitten", "sitting") == pytest.approx(1 - 3 / 7)
    assert Levenshtein()("", "") == 1.0 and Levenshtein()("a", None) is None
    assert NumericCloseness()(9, 10) == pytest.approx(0.9) and NumericCloseness()("x", 1) == 0.0
    assert NumericCloseness()(1, None) is None
    assert RegexMatch(r"\d{3}")("call 555") == 1.0 and RegexMatch("z")("a") == 0.0


def test_llm_judge_sync_async_and_unparseable():
    judge = LLMJudge(lambda p: "looks right\nChoice: A", rubric="correct?")
    s = asyncio.run(judge(input="q", output="a", expected="a"))
    assert s.score == 1.0 and s.metadata["choice"] == "A"

    async def allm(prompt):
        assert "correct?" in prompt and "Choice: X" in prompt
        return "meh\nChoice: B"

    assert asyncio.run(LLMJudge(allm, rubric="correct?")(input="q", output="a")).score == 0.5
    bad = asyncio.run(LLMJudge(lambda p: "no idea", rubric="r")(input="q", output="a"))
    assert bad.score is None and "unparseable" in bad.metadata["error"]


def test_llm_judge_inside_eval(tmp_path):
    r = ew.Eval("judge", DATA, str.upper, [LLMJudge(lambda p: "Choice: C", rubric="r")], store_dir=tmp_path)
    assert r.score("llm_judge") == 0.0


def test_experiment_ordering_within_same_second(tmp_path):
    ids = [ew.Eval("fast", DATA, str.upper, [ExactMatch()], store_dir=tmp_path).experiment_id for _ in range(6)]
    from evalwright.evals import list_experiments
    assert list_experiments(tmp_path, "fast") == ids                      # creation order, not id order
    assert load_experiment(tmp_path, "fast", "previous")["id"] == ids[-2]
    assert load_experiment(tmp_path, "fast", ids[0])["id"] == ids[0]
    assert load_experiment(tmp_path, "fast", "nope") is None
