import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { compareExperiments, Eval, loadExperiment, scorers } from "../src/index.js";
import { assertEval, captureSpans, type SpanCapture } from "../src/testing.js";

let cap: SpanCapture | null = null;
afterEach(() => { cap?.restore(); cap = null; });
const store = () => mkdtempSync(join(tmpdir(), "ew-evals-"));
const data = Array.from({ length: 10 }, (_, i) => ({ input: i, expected: i * 2 }));

describe("Eval", () => {
  it("runs cases with bounded concurrency and summarizes", async () => {
    let active = 0;
    let peak = 0;
    const res = await Eval("double", {
      data,
      task: async (x: number) => {
        peak = Math.max(peak, ++active);
        await new Promise((r) => setTimeout(r, 5));
        active--;
        return x * 2;
      },
      scores: [scorers.exactMatch(), scorers.numericCloseness()],
      maxConcurrency: 3,
      storeDir: store(),
    });
    expect(peak).toBeLessThanOrEqual(3);
    expect(res.summary).toMatchObject({ n_cases: 10, n_runs: 10, errors: 0, error_rate: 0 });
    expect(res.score("exact_match")).toBe(1);
    expect(res.summary.scores.exact_match!.ci95).toEqual([1, 1]);
    expect(res.toString()).toContain("| exact_match | 1.000 |");
  });

  it("records task errors, timeouts and scorer errors without aborting", async () => {
    const throwing = Object.assign(() => { throw new Error("scorer bug"); }, { scorerName: "fragile" });
    const res = await Eval("mixed", {
      data: [{ input: "ok" }, { input: "boom" }, { input: "slow" }],
      task: async (x: string) => {
        if (x === "boom") throw new SyntaxError("bad input");
        if (x === "slow") await new Promise((r) => setTimeout(r, 200));
        return x;
      },
      scores: [throwing, () => 1],
      timeoutMs: 50,
      storeDir: null,
    });
    const byInput = Object.fromEntries(res.cases.map((c) => [c.input as string, c]));
    expect(byInput.boom!.error).toEqual({ type: "SyntaxError", message: "bad input" });
    expect(byInput.slow!.error?.type).toBe("Timeout");
    expect(byInput.ok!.scores.fragile).toBeNull();
    expect(byInput.ok!.scorer_errors?.fragile).toMatch(/scorer bug/);
    expect(res.summary.error_rate).toBeCloseTo(2 / 3);
    expect(res.experimentId).toBeNull();
  });

  it("supports trials, plain inputs, data factories, Score objects and dict results", async () => {
    const res = await Eval("shapes", {
      data: async () => ["a", "bb"],
      task: (s: string) => s.length,
      scores: [
        Object.assign(() => ({ name: "custom", score: 0.5, metadata: { why: "test" } }), { scorerName: "custom" }),
        () => ({ p: 1, q: null }),
        () => true,
      ],
      trials: 3,
      storeDir: null,
    });
    expect(res.summary).toMatchObject({ n_cases: 2, trials: 3, n_runs: 6 });
    expect(res.score("custom")).toBe(0.5);
    expect(res.cases[0]!.score_metadata.custom).toEqual({ why: "test" });
    expect(res.score("p")).toBe(1);
    expect(res.summary.scores.q!.n).toBe(0);
  });

  it("out-of-range scores become scorer errors", async () => {
    const res = await Eval("range", { data: [{ input: 1 }], task: (x: number) => x, scores: [() => 1.5], storeDir: null });
    expect(Object.values(res.cases[0]!.scorer_errors ?? {})[0]).toMatch(/outside \[0, 1\]/);
  });

  it("stores experiments, compares with the previous run, and gates regressions", async () => {
    const dir = store();
    const first = await Eval("gate", { data, task: (x: number) => x * 2, scores: [scorers.exactMatch()], storeDir: dir });
    expect(first.comparison).toBeNull();
    const second = await Eval("gate", { data, task: (x: number) => (x < 5 ? x * 2 : 0), scores: [scorers.exactMatch()], storeDir: dir });
    expect(second.comparison).toMatchObject({ baseline: first.experimentId, common_cases: 10 });
    expect(second.comparison!.scores.exact_match).toMatchObject({ diff: -0.5, regressed: 5, improved: 0 });
    expect(() => assertEval(second, { maxRegression: 0.1 })).toThrow(/regressed by 0.5000/);
    expect(() => assertEval(second, { min: { exact_match: 0.9 } })).toThrow(/below minimum/);
    expect(assertEval(first, { min: { exact_match: 1 } })).toBe(first);
    expect(second.check({ maxErrorRate: 0 })).toEqual([]);

    // on-disk layout is the Python SDK's: cases.jsonl + summary.json + LATEST pointer
    const latest = readFileSync(join(dir, "gate", "LATEST"), "utf8");
    expect(latest).toBe(second.experimentId);
    const loaded = await loadExperiment(dir, "gate", "previous");
    expect(loaded?.id).toBe(first.experimentId);
    expect(compareExperiments(loaded!, { id: "x", cases: second.cases }).scores.exact_match!.diff).toBe(-0.5);
    expect(await loadExperiment(dir, "nope")).toBeNull();
  });

  it("every case is an eval span with task and score children", async () => {
    cap = captureSpans();
    await Eval("traced-eval", { data: [{ input: 1, expected: 2 }], task: (x: number) => x * 2, scores: [scorers.exactMatch()], storeDir: null });
    expect(await cap.tree()).toBe("traced-eval [eval]\n  task [task]\n  exact_match [score]");
    const root = await cap.find("traced-eval");
    expect(root.scores).toEqual({ exact_match: 1 });
    expect(root.metadata).toMatchObject({ trial: 0 });
  });

  it("empty data is an error", async () => {
    await expect(Eval("empty", { data: [], task: (x: unknown) => x })).rejects.toThrow(/no cases/);
  });
});

describe("scorers", () => {
  const run = (s: ReturnType<typeof scorers.exactMatch>, output: unknown, expected?: unknown) =>
    s({ input: null, output, expected, metadata: {} });

  it("exactMatch / contains / jsonValid / regexMatch", () => {
    expect(run(scorers.exactMatch(), " Hi ", "Hi")).toBe(1);
    expect(run(scorers.exactMatch({ caseSensitive: false }), "HI", "hi")).toBe(1);
    expect(run(scorers.exactMatch(), { b: 1, a: 2 }, { a: 2, b: 1 })).toBe(1);     // key order irrelevant
    expect(run(scorers.exactMatch(), "x")).toBeNull();
    expect(run(scorers.contains(), "The Quick fox", ["quick", "fox"])).toBe(1);
    expect(run(scorers.contains({ mode: "fraction" }), "a b", ["a", "z"])).toBe(0.5);
    expect(run(scorers.contains({ mode: "any" }), "a b", ["z", "b"])).toBe(1);
    expect(run(scorers.contains(), "a", [])).toBeNull();
    expect(run(scorers.jsonValid(), '{"a":1}')).toBe(1);
    expect(run(scorers.jsonValid(), "{nope")).toBe(0);
    expect(run(scorers.jsonValid({ requiredKeys: ["a", "b"] }), { a: 1 })).toBe(0);
    expect(run(scorers.regexMatch(/^\d+$/g), "123")).toBe(1);
    expect(run(scorers.regexMatch(/^\d+$/g), "123")).toBe(1);   // lastIndex reset with /g
  });

  it("levenshtein and numericCloseness", () => {
    expect(run(scorers.levenshtein(), "kitten", "sitting")).toBeCloseTo(1 - 3 / 7);
    expect(run(scorers.levenshtein(), "", "")).toBe(1);
    expect(run(scorers.numericCloseness(), 9, 10)).toBeCloseTo(0.9);
    expect(run(scorers.numericCloseness(), "nan", 10)).toBe(0);
  });

  it("llmJudge parses the final choice and is model-agnostic", async () => {
    let prompt = "";
    const judge = scorers.llmJudge(async (p) => { prompt = p; return "Looks right.\nChoice: B\nChoice: A"; }, { rubric: "Is it right?" });
    const s = (await judge({ input: "2+2", output: "4", expected: "4", metadata: {} })) as unknown as { score: number; metadata: { choice: string } };
    expect(s.score).toBe(1);
    expect(s.metadata.choice).toBe("A");
    expect(prompt).toContain("Criteria: Is it right?");
    const bad = scorers.llmJudge(() => "no idea", { rubric: "x" });
    expect(await bad({ input: 1, output: 2, expected: undefined, metadata: {} })).toMatchObject({ score: null });
  });
});
