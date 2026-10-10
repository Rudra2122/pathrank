/**
 * Test helpers (import from "evalwright/testing"). Framework-agnostic: works with vitest, jest,
 * node:test or anything that treats a thrown Error as a failure.
 *
 *   const cap = captureSpans();
 *   await runAgent("hi");
 *   expect(await cap.tree()).toBe("agent [task]\n  draft [llm]");
 *   cap.restore();
 *
 *   assertEval(await Eval("agent", {...}), { min: { judge: 0.8 }, maxRegression: 0.02 });
 */
import type { EvalResult, GateOptions } from "./evals.js";
import { InMemoryExporter } from "./exporters.js";
import { BatchProcessor } from "./processor.js";
import type { SpanRecord } from "./span.js";
import { getTracer, setTracer, Tracer } from "./tracer.js";

export class SpanCapture {
  readonly #exporter: InMemoryExporter;
  readonly #previous: Tracer;
  readonly tracer: Tracer;

  constructor(project = "test", opts: { sampleRate?: number } = {}) {
    this.#exporter = new InMemoryExporter();
    this.#previous = getTracer();
    this.tracer = new Tracer({
      project,
      processor: new BatchProcessor(this.#exporter, { flushIntervalMs: 10, exitHook: false }),
      ...(opts.sampleRate !== undefined ? { sampleRate: opts.sampleRate } : {}),
    });
    setTracer(this.tracer);
  }

  /** All spans recorded so far (flushes first). */
  async spans(): Promise<SpanRecord[]> {
    await this.tracer.flush(5000);
    return [...this.#exporter.spans];
  }

  async names(): Promise<string[]> {
    return (await this.spans()).map((s) => s.name);
  }

  async find(name: string): Promise<SpanRecord> {
    const spans = await this.spans();
    const s = spans.find((x) => x.name === name);
    if (!s) throw new Error(`no span named ${JSON.stringify(name)}; recorded: ${[...new Set(spans.map((x) => x.name))].sort().join(", ")}`);
    return s;
  }

  async children(span: SpanRecord): Promise<SpanRecord[]> {
    return (await this.spans()).filter((s) => s.parent_id === span.span_id).sort((a, b) => a.start_ns - b.start_ns);
  }

  /** Indented `name [type]` per span, children in start order. Same format as the Python SDK. */
  async tree(o: { showStatus?: boolean } = {}): Promise<string> {
    const spans = await this.spans();
    const ids = new Set(spans.map((s) => s.span_id));
    const byParent = new Map<string | null, SpanRecord[]>();
    for (const s of spans) {
      const p = s.parent_id !== null && ids.has(s.parent_id) ? s.parent_id : null;
      byParent.set(p, [...(byParent.get(p) ?? []), s]);
    }
    const lines: string[] = [];
    const walk = (parent: string | null, depth: number) => {
      for (const s of [...(byParent.get(parent) ?? [])].sort((a, b) => a.start_ns - b.start_ns)) {
        lines.push(`${"  ".repeat(depth)}${s.name} [${s.type}]${o.showStatus ? ` ${s.status}` : ""}`);
        walk(s.span_id, depth + 1);
      }
    };
    walk(null, 0);
    return lines.join("\n");
  }

  clear(): void {
    this.#exporter.spans.length = 0;
  }

  /** Reinstall the tracer that was active before capture. */
  restore(): void {
    setTracer(this.#previous);
  }
}

/** Route all tracing to memory until `restore()` is called. */
export function captureSpans(project = "test", opts: { sampleRate?: number } = {}): SpanCapture {
  return new SpanCapture(project, opts);
}

/** Throw (with the summary) if any gate fails; return the result otherwise. */
export function assertEval(result: EvalResult, gates: GateOptions): EvalResult {
  const fails = result.check(gates);
  if (fails.length) throw new Error(`eval gates failed:\n  ${fails.join("\n  ")}\n\n${result.toMarkdown()}`);
  return result;
}
