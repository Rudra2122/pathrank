import { afterEach, describe, expect, it } from "vitest";
import * as ew from "../src/index.js";
import { captureSpans, type SpanCapture } from "../src/testing.js";

let cap: SpanCapture | null = null;
afterEach(() => { cap?.restore(); cap = null; });

const tick = (ms = 1) => new Promise((r) => setTimeout(r, ms));

describe("spans and context", () => {
  it("nests sync and async spans and records data", async () => {
    cap = captureSpans();
    const step = ew.traced((x: number) => x + 1, { name: "step", type: "tool" });
    const llm = ew.traced(async (x: number) => { await tick(); ew.log({ metrics: { tokens: 3 } }); return x * 2; }, { name: "llm", type: "llm" });
    const out = await ew.span("root", { type: "task", input: "q" }, async () => llm(step(1)));
    expect(out).toBe(4);
    expect(await cap.tree()).toBe("root [task]\n  step [tool]\n  llm [llm]");
    const l = await cap.find("llm");
    expect(l.metrics.tokens).toBe(3);
    expect(l.input).toBe(2);          // one argument is logged as-is
    expect(l.output).toBe(4);
    const root = await cap.find("root");
    expect(l.trace_id).toBe(root.trace_id);
    expect(root.input).toBe("q");
  });

  it("concurrent async work never mixes spans (AsyncLocalStorage)", async () => {
    cap = captureSpans();
    const leaf = ew.traced(async (i: number) => { await tick(Math.random() * 5); return i; }, { name: "leaf" });
    await Promise.all([0, 1, 2, 3, 4].map((i) => ew.span(`parent${i}`, async () => { await tick(Math.random() * 5); await leaf(i); })));
    const spans = await cap.spans();
    for (const s of spans.filter((x) => x.name === "leaf")) {
      const parent = spans.find((p) => p.span_id === s.parent_id)!;
      expect(parent.name).toBe(`parent${s.input}`);
    }
  });

  it("records errors and rethrows them (sync and async)", async () => {
    cap = captureSpans();
    expect(() => ew.span("bad", () => { throw new RangeError("nope"); })).toThrow("nope");
    await expect(ew.span("bad2", async () => { throw new TypeError("async nope"); })).rejects.toThrow("async nope");
    const b = await cap.find("bad");
    expect(b.status).toBe("error");
    expect(b.error).toMatchObject({ type: "RangeError", message: "nope" });
    expect((await cap.find("bad2")).error?.type).toBe("TypeError");
  });

  it("generators hold the span only while running, never while suspended", async () => {
    cap = captureSpans();
    const gen = ew.traced(function* count(n: number) { for (let i = 0; i < n; i++) yield i; }, { name: "gen" });
    const seenOutside: (string | undefined)[] = [];
    for (const _ of gen(3)) seenOutside.push(ew.currentSpan()?.name);
    expect(seenOutside).toEqual([undefined, undefined, undefined]);

    const agen = ew.traced(async function* stream() { yield "a"; await tick(); yield "b"; }, { name: "agen" });
    const got: string[] = [];
    for await (const x of agen()) got.push(x);
    expect(got).toEqual(["a", "b"]);
    expect((await cap.find("gen")).metrics.items).toBe(3);
    expect((await cap.find("agen")).metrics.items).toBe(2);
  });

  it("a generator abandoned early still ends its span", async () => {
    cap = captureSpans();
    const gen = ew.traced(function* inf() { let i = 0; while (true) yield i++; }, { name: "inf" });
    for (const x of gen()) if (x === 2) break;
    expect((await cap.find("inf")).metrics.items).toBe(3);
  });

  it("generator errors are recorded and rethrown", async () => {
    cap = captureSpans();
    const gen = ew.traced(function* g() { yield 1; throw new Error("gen fail"); }, { name: "g" });
    expect(() => [...gen()]).toThrow("gen fail");
    expect((await cap.find("g")).status).toBe("error");
  });

  it("wrapping preserves `this` and function name", async () => {
    cap = captureSpans();
    class Bot { prefix = "hi "; greet = ew.traced(function (this: Bot, n: string) { return this.prefix + n; }, { name: "greet" }); }
    expect(new Bot().greet("x")).toBe("hi x");
    const f = ew.traced(function namedFn() { return 1; });
    expect(f.name).toBe("namedFn");
  });

  it("captureArgs / captureOutput can be turned off", async () => {
    cap = captureSpans();
    const f = ew.traced((secret: string) => secret.length, { name: "f", captureArgs: false, captureOutput: false });
    f("hidden");
    const s = await cap.find("f");
    expect(s.input).toBeNull();
    expect(s.output).toBeNull();
  });

  it("explicit parent and new traces", async () => {
    cap = captureSpans();
    await ew.span("outer", async () => {
      ew.span("detached", { parent: null }, () => undefined);
    });
    const [o, d] = [await cap.find("outer"), await cap.find("detached")];
    expect(d.parent_id).toBeNull();
    expect(d.trace_id).not.toBe(o.trace_id);
  });

  it("manual lifecycle with startSpan; end() is idempotent", async () => {
    cap = captureSpans();
    const s = cap.tracer.startSpan("manual", { type: "http", tags: ["a"] });
    s.log({ tags: ["a", "b"], scores: { ok: 1 } });
    s.end();
    s.end();
    const spans = await cap.spans();
    expect(spans.filter((x) => x.name === "manual")).toHaveLength(1);
    expect(spans[0]!.tags).toEqual(["a", "b"]);
    expect(s.ended).toBe(true);
  });
});

describe("sampling and disabled tracers", () => {
  it("sampling is per trace: whole traces are kept or dropped", async () => {
    cap = captureSpans("t", { sampleRate: 0.5 });
    for (let i = 0; i < 200; i++) ew.span("root", () => ew.span("child", () => undefined));
    const spans = await cap.spans();
    const roots = spans.filter((s) => s.name === "root");
    const children = spans.filter((s) => s.name === "child");
    expect(roots.length).toBeGreaterThan(50);
    expect(roots.length).toBeLessThan(150);
    expect(children.length).toBe(roots.length);
    const rootIds = new Set(roots.map((r) => r.span_id));
    expect(children.every((c) => rootIds.has(c.parent_id!))).toBe(true);
  });

  it("sampleRate 0 records nothing, and logging into sampled-out spans is a no-op", async () => {
    cap = captureSpans("t", { sampleRate: 0 });
    ew.span("x", (s) => { s.log({ output: 1 }); expect(s.sampled).toBe(false); });
    expect(await cap.spans()).toEqual([]);
  });

  it("without init everything works and nothing is recorded", async () => {
    const t = new ew.Tracer({ enabled: false });
    expect(t.span("x", () => 5)).toBe(5);
    expect(await t.flush()).toBe(true);
    expect(t.startSpan("y")).toBe(ew.NOOP_SPAN);
  });

  it("functions wrapped before init() report to the tracer installed later", async () => {
    const f = ew.traced(() => 1, { name: "early" });
    cap = captureSpans();
    f();
    expect(await cap.names()).toEqual(["early"]);
  });

  it("rejects invalid sample rates", () => {
    expect(() => new ew.Tracer({ sampleRate: 2 })).toThrow(RangeError);
  });
});

describe("init()", () => {
  it("reads EVALWRIGHT_* environment variables", async () => {
    const old = { ...process.env };
    try {
      process.env.EVALWRIGHT_EXPORT = "memory";
      process.env.EVALWRIGHT_SAMPLE_RATE = "0.25";
      const t = ew.init("envproj");
      expect(t.project).toBe("envproj");
      expect(t.sampleRate).toBe(0.25);
      expect(t.processor?.exporter).toBeInstanceOf(ew.InMemoryExporter);
      process.env.EVALWRIGHT_DISABLED = "1";
      expect(ew.init("off").enabled).toBe(false);
      process.env.EVALWRIGHT_DISABLED = "";
      process.env.EVALWRIGHT_EXPORT = "http:";
      expect(() => ew.init("x")).toThrow(/needs a URL/);
    } finally {
      process.env = old;
      ew.setTracer(new ew.Tracer({ enabled: false }));
    }
  });
});
