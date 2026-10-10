import { newSpanId, newTraceId } from "./ids.js";
import { type Json, type Limits, type Redactor, toJsonable } from "./serialize.js";

export type SpanType = "task" | "llm" | "tool" | "function" | "eval" | "score" | "http" | "retrieval" | "custom";

/** One finished span on the wire (sdk/spec/span.schema.json). */
export interface SpanRecord {
  trace_id: string;
  span_id: string;
  parent_id: string | null;
  name: string;
  type: SpanType;
  start_ns: number;
  end_ns: number;
  duration_ms: number;
  cpu_ms: number | null;
  input: Json;
  output: Json;
  expected: Json;
  metadata: { [k: string]: Json };
  metrics: { [k: string]: number };
  scores: { [k: string]: number | null };
  tags: string[];
  status: "ok" | "error";
  error: { type: string; message: string; stack: string } | null;
  project: string;
}

/** Fields accepted by `span.log()`. `undefined` means "leave unchanged"; later calls merge. */
export interface LogFields {
  input?: unknown;
  output?: unknown;
  expected?: unknown;
  metadata?: Record<string, unknown>;
  metrics?: Record<string, number>;
  scores?: Record<string, number | null>;
  tags?: string[];
}

export interface SpanSink {
  onEnd(span: Span): void;
}

const epochNs = (): number => Math.round((performance.timeOrigin + performance.now()) * 1e6);

export class Span {
  readonly traceId: string;
  readonly spanId: string;
  readonly parentId: string | null;
  readonly name: string;
  readonly type: SpanType;
  readonly sampled: boolean;
  readonly startNs: number;
  endNs: number | null = null;
  durationMs: number | null = null;
  input: unknown = null;
  output: unknown = null;
  expected: unknown = null;
  metadata: Record<string, unknown> = {};
  metrics: Record<string, number> = {};
  scores: Record<string, number | null> = {};
  tags: string[] = [];
  status: "ok" | "error" = "ok";
  error: { type: string; message: string; stack: string } | null = null;
  #t0: number;
  #ended = false;
  #sink: SpanSink | null;

  constructor(opts: { name: string; type?: SpanType; parent?: Span | null; sampled?: boolean; sink?: SpanSink | null; traceId?: string }) {
    const p = opts.parent ?? null;
    this.name = opts.name;
    this.type = opts.type ?? "function";
    this.traceId = p ? p.traceId : (opts.traceId ?? newTraceId());
    this.parentId = p ? p.spanId : null;
    this.spanId = newSpanId();
    this.sampled = p ? p.sampled : (opts.sampled ?? true);
    this.#sink = opts.sink ?? null;
    this.startNs = epochNs();
    this.#t0 = performance.now();
  }

  /** Attach data to the span. Returns the span for chaining. */
  log(f: LogFields): this {
    if (!this.sampled) return this;
    if (f.input !== undefined) this.input = f.input;
    if (f.output !== undefined) this.output = f.output;
    if (f.expected !== undefined) this.expected = f.expected;
    if (f.metadata) Object.assign(this.metadata, f.metadata);
    if (f.metrics) for (const [k, v] of Object.entries(f.metrics)) this.metrics[k] = Number(v);
    if (f.scores) Object.assign(this.scores, f.scores);
    if (f.tags) for (const t of f.tags) if (!this.tags.includes(t)) this.tags.push(t);
    return this;
  }

  setError(err: unknown): void {
    this.status = "error";
    if (!this.sampled) return;
    const e = err instanceof Error ? err : new Error(String(err));
    this.error = { type: e.name || "Error", message: e.message.slice(0, 2000), stack: (e.stack ?? "").slice(-4000) };
  }

  end(): void {
    if (this.#ended) return;
    this.#ended = true;
    this.durationMs = performance.now() - this.#t0;
    this.endNs = Math.max(this.startNs, this.startNs + Math.round(this.durationMs * 1e6));
    if (this.sampled && this.#sink) this.#sink.onEnd(this);
  }

  get ended(): boolean {
    return this.#ended;
  }

  toRecord(project: string, limits: Limits, redactor?: Redactor): SpanRecord {
    const j = (v: unknown): Json => (v === null || v === undefined ? null : toJsonable(v, limits, redactor));
    const metrics: Record<string, number> = {};
    for (const [k, v] of Object.entries(this.metrics)) if (Number.isFinite(v)) metrics[k] = v;
    return {
      trace_id: this.traceId,
      span_id: this.spanId,
      parent_id: this.parentId,
      name: this.name,
      type: this.type,
      start_ns: this.startNs,
      end_ns: this.endNs ?? this.startNs,
      duration_ms: this.durationMs ?? 0,
      cpu_ms: null,
      input: j(this.input),
      output: j(this.output),
      expected: j(this.expected),
      metadata: toJsonable(this.metadata, limits, redactor) as { [k: string]: Json },
      metrics,
      scores: { ...this.scores },
      tags: [...this.tags],
      status: this.status,
      error: this.error,
      project,
    };
  }
}

/** Shared stand-in for disabled tracers and sampled-out traces. */
class NoopSpan extends Span {
  constructor() {
    super({ name: "noop", type: "custom", sampled: false, traceId: "0".repeat(32) });
  }
  override log(): this {
    return this;
  }
  override setError(): void {}
  override end(): void {}
}

export const NOOP_SPAN: Span = new NoopSpan();
