import { createServer, type IncomingMessage, type Server } from "node:http";
import { gunzipSync } from "node:zlib";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  BatchProcessor, CircuitBreaker, ConsoleExporter, type Exporter, ExportResult, HttpExporter, JsonlExporter,
  OtlpHttpExporter, RetryPolicy, type SpanRecord, Tracer,
} from "../src/index.js";
import { parseRetryAfter } from "../src/exporters.js";

const fastRetry = (maxRetries = 5) => new RetryPolicy({ maxRetries, baseDelayMs: 1, maxDelayMs: 5 });
const rec = (i: number): SpanRecord => {
  const t = new Tracer({ project: "p" });
  const s = t.startSpan(`s${i}`);
  s.end();
  return s.toRecord("p", t.limits);
};

class ScriptedExporter implements Exporter {
  calls = 0;
  delivered: SpanRecord[] = [];
  constructor(private script: (call: number) => ExportResult | "throw") {}
  async export(batch: SpanRecord[]): Promise<ExportResult> {
    const r = this.script(this.calls++);
    if (r === "throw") throw new Error("exporter exploded");
    if (r.success) this.delivered.push(...batch);
    return r;
  }
}

describe("BatchProcessor", () => {
  it("retries transient failures and delivers everything", async () => {
    const exp = new ScriptedExporter((c) => (c < 3 ? ExportResult.retry("503") : ExportResult.ok()));
    const p = new BatchProcessor(exp, { retry: fastRetry(), exitHook: false });
    for (let i = 0; i < 5; i++) p.onEnd(rec(i));
    expect(await p.flush(2000)).toBe(true);
    expect(exp.delivered).toHaveLength(5);
    expect(p.stats()).toMatchObject({ exported: 5, retries: 3, dropped_failed: 0, circuit: "closed" });
    await p.shutdown();
  });

  it("a throwing exporter is treated as retryable and never escapes", async () => {
    const exp = new ScriptedExporter((c) => (c === 0 ? "throw" : ExportResult.ok()));
    const p = new BatchProcessor(exp, { retry: fastRetry(), exitHook: false });
    p.onEnd(rec(1));
    expect(await p.flush(2000)).toBe(true);
    expect(p.stats().last_error).toMatch(/exporter exploded/);
    await p.shutdown();
  });

  it("non-retryable failures drop the batch once, without retries", async () => {
    const exp = new ScriptedExporter(() => ExportResult.fail("HTTP 400"));
    const p = new BatchProcessor(exp, { retry: fastRetry(), exitHook: false });
    p.onEnd(rec(1));
    expect(await p.flush(2000)).toBe(false);
    expect(exp.calls).toBe(1);
    expect(p.stats()).toMatchObject({ dropped_failed: 1, batches_failed: 1, retries: 0 });
    await p.shutdown();
  });

  it("gives up after maxRetries and counts the drop", async () => {
    const exp = new ScriptedExporter(() => ExportResult.retry("down"));
    const p = new BatchProcessor(exp, { retry: fastRetry(2), breaker: new CircuitBreaker(100), exitHook: false });
    p.onEnd(rec(1));
    expect(await p.flush(2000)).toBe(false);
    expect(exp.calls).toBe(3);
    await p.shutdown();
  });

  it("bounded queue drops the oldest spans and counts them", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    const exp: Exporter = { export: async () => { await gate; return ExportResult.ok(); } };
    const p = new BatchProcessor(exp, { maxQueue: 3, maxBatch: 1, flushIntervalMs: 1, exitHook: false });
    p.onEnd(rec(0));                       // taken by the (blocked) export loop
    await new Promise((r) => setTimeout(r, 10));
    for (let i = 1; i <= 6; i++) p.onEnd(rec(i));
    expect(p.stats()).toMatchObject({ queued: 3, dropped_queue_full: 3 });
    release();
    await p.shutdown();
  });

  it("circuit breaker opens, pauses exports, then probes and recovers", async () => {
    let now = 0;
    const breaker = new CircuitBreaker(2, 1000, () => now);
    let healthy = false;
    const exp = new ScriptedExporter(() => (healthy ? ExportResult.ok() : ExportResult.retry("down")));
    const p = new BatchProcessor(exp, { retry: fastRetry(10), breaker, exitHook: false });
    p.onEnd(rec(1));
    expect(await p.flush(500)).toBe(false);
    expect(breaker.state).toBe("open");
    const callsWhileOpen = exp.calls;
    p.onEnd(rec(2));
    expect(await p.flush(100)).toBe(false);          // open: no export attempts
    expect(exp.calls).toBe(callsWhileOpen);
    now += 1001;
    expect(breaker.state).toBe("half_open");
    healthy = true;
    expect(await p.flush(2000)).toBe(true);           // probe succeeds -> closed
    expect(breaker.state).toBe("closed");
    await p.shutdown();
  });

  it("flush times out instead of hanging", async () => {
    const exp: Exporter = { export: () => new Promise(() => undefined) };    // never settles
    const p = new BatchProcessor(exp, { exitHook: false });
    p.onEnd(rec(1));
    const t0 = performance.now();
    expect(await p.flush(100)).toBe(false);
    expect(performance.now() - t0).toBeLessThan(1000);
  });

  it("validates options and backoff math", () => {
    expect(() => new BatchProcessor(new ConsoleExporter(), { maxBatch: 0 })).toThrow(RangeError);
    const r = new RetryPolicy({ baseDelayMs: 100, maxDelayMs: 1000 });
    expect(r.delayMs(10, undefined, () => 1)).toBe(1000);       // capped
    expect(r.delayMs(0, 50_000, () => 1)).toBe(1000);           // Retry-After capped
    expect(r.delayMs(2, undefined, () => 0.5)).toBe(200);       // full jitter
    expect(parseRetryAfter("2")).toBe(2000);
    expect(parseRetryAfter(null)).toBeUndefined();
    expect(parseRetryAfter("garbage")).toBeUndefined();
    expect(parseRetryAfter(new Date(Date.now() + 60_000).toUTCString())).toBeGreaterThan(50_000);
  });
});

// ---- HTTP exporter against a real server -------------------------------------------------------
let server: Server | null = null;
afterEach(() => { server?.close(); server = null; });

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((r) => { const parts: Buffer[] = []; req.on("data", (c: Buffer) => parts.push(c)); req.on("end", () => r(Buffer.concat(parts))); });
}

async function collector(failFirst: number, status = 503, retryAfter?: string) {
  const received: SpanRecord[] = [];
  const keys: string[] = [];
  const headers: Record<string, string | undefined>[] = [];
  let n = 0;
  server = createServer(async (req, res) => {
    const raw = await readBody(req);
    headers.push({ ua: req.headers["user-agent"], enc: req.headers["content-encoding"], auth: req.headers.authorization });
    const key = String(req.headers["idempotency-key"]);
    keys.push(key);
    if (n++ < failFirst) {
      res.writeHead(status, retryAfter ? { "Retry-After": retryAfter } : {});
      return res.end("{}");
    }
    const body = req.headers["content-encoding"] === "gzip" ? gunzipSync(raw) : raw;
    received.push(...(JSON.parse(body.toString()) as { spans: SpanRecord[] }).spans);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end('{"accepted":1}');
  });
  await new Promise<void>((r) => server!.listen(0, "127.0.0.1", () => r()));
  const port = (server!.address() as { port: number }).port;
  return { url: `http://127.0.0.1:${port}/v1/spans`, received, keys, headers };
}

describe("HttpExporter", () => {
  it("delivers through 503s with the same idempotency key on every retry", async () => {
    const c = await collector(2, 503, "0");
    const p = new BatchProcessor(new HttpExporter(c.url, { apiKey: "k" }), { retry: fastRetry(), exitHook: false });
    p.onEnd(rec(1));
    p.onEnd(rec(2));
    expect(await p.flush(5000)).toBe(true);
    expect(c.received).toHaveLength(2);
    expect(new Set(c.keys).size).toBe(1);                  // retries are recognizable as duplicates
    expect(c.keys[0]).toMatch(/^[0-9a-f]{32}$/);
    expect(c.headers[0]).toMatchObject({ enc: "gzip", auth: "Bearer k" });
    expect(c.headers[0]!.ua).toMatch(/^evalwright-typescript\/\d+\.\d+\.\d+ node\//);
    await p.shutdown();
  });

  it("4xx is not retried", async () => {
    const c = await collector(100, 401);
    const res = await new HttpExporter(c.url).export([rec(1)]);
    expect(res).toMatchObject({ success: false, retryable: false, error: "HTTP 401" });
  });

  it("429 with Retry-After is retryable with the server's delay", async () => {
    const c = await collector(100, 429, "3");
    const res = await new HttpExporter(c.url, { gzip: false }).export([rec(1)]);
    expect(res).toMatchObject({ retryable: true, retryAfterMs: 3000 });
  });

  it("network errors and timeouts are retryable", async () => {
    const down = await new HttpExporter("http://127.0.0.1:1/v1/spans", { timeoutMs: 500 }).export([rec(1)]);
    expect(down.retryable).toBe(true);
    server = createServer(() => undefined);                 // accepts, never answers
    await new Promise<void>((r) => server!.listen(0, "127.0.0.1", () => r()));
    const port = (server!.address() as { port: number }).port;
    const slow = await new HttpExporter(`http://127.0.0.1:${port}/`, { timeoutMs: 100 }).export([rec(1)]);
    expect(slow.retryable).toBe(true);
    server.closeAllConnections();
  });

  it("OTLP payload is valid OpenTelemetry JSON", () => {
    const t = new Tracer({ project: "p" });
    const parent = t.startSpan("parent", { type: "task" });
    const child = t.startSpan("child", { type: "llm", parent });
    child.log({ metrics: { tokens: 5 }, scores: { acc: 1 }, metadata: { m: 1 }, output: "x" });
    child.setError(new Error("bad"));
    child.end();
    parent.end();
    const body = new OtlpHttpExporter("http://x", "svc").payload([parent.toRecord("p", t.limits), child.toRecord("p", t.limits)]) as {
      resourceSpans: { resource: { attributes: { key: string }[] }; scopeSpans: { spans: Record<string, unknown>[] }[] }[];
    };
    const spans = body.resourceSpans[0]!.scopeSpans[0]!.spans;
    expect(body.resourceSpans[0]!.resource.attributes[0]!.key).toBe("service.name");
    expect(spans[1]).toMatchObject({ parentSpanId: parent.spanId, status: { code: 2, message: "bad" } });
    expect(String(spans[0]!.startTimeUnixNano)).toMatch(/^\d{16,}$/);
  });
});

describe("other exporters", () => {
  it("JSONL appends one record per line, creating directories", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ew-"));
    const path = join(dir, "nested", "t.jsonl");
    const e = new JsonlExporter(path);
    await e.export([rec(1), rec(2)]);
    await e.export([rec(3)]);
    expect(readFileSync(path, "utf8").trim().split("\n")).toHaveLength(3);
  });

  it("console exporter writes a line per span", async () => {
    const writes: string[] = [];
    const orig = process.stderr.write.bind(process.stderr);
    process.stderr.write = ((s: string) => { writes.push(s); return true; }) as typeof process.stderr.write;
    try {
      await new ConsoleExporter().export([rec(1)]);
    } finally {
      process.stderr.write = orig;
    }
    expect(writes[0]).toMatch(/\[evalwright\] function +s1/);
  });
});
