import type { AsyncLocalStorage } from "node:async_hooks";
import { join } from "node:path";
import {
  ConsoleExporter, type Exporter, HttpExporter, InMemoryExporter, JsonlExporter, OtlpHttpExporter,
} from "./exporters.js";
import { newTraceId, sampled as sampleDecision } from "./ids.js";
import { BatchProcessor, type BatchProcessorOptions } from "./processor.js";
import { globalState } from "./global.js";
import { DEFAULT_LIMITS, defaultRedactor, type Limits, type Redactor } from "./serialize.js";
import { type LogFields, NOOP_SPAN, Span, type SpanType } from "./span.js";

const storage = globalState().storage as AsyncLocalStorage<Span>;   // shared across package copies

export interface SpanOptions {
  type?: SpanType;
  input?: unknown;
  expected?: unknown;
  metadata?: Record<string, unknown>;
  tags?: string[];
  /** Explicit parent; defaults to the current span. `null` starts a new trace. */
  parent?: Span | null;
}

export interface TracedOptions {
  name?: string;
  type?: SpanType;
  /** Log the call's arguments as input (default: tracer setting, true). One argument is logged as-is. */
  captureArgs?: boolean;
  captureOutput?: boolean;
}

export interface TracerOptions {
  project?: string;
  processor?: BatchProcessor | null;
  sampleRate?: number;
  redactor?: Redactor;
  limits?: Limits;
  enabled?: boolean;
  captureArgs?: boolean;
}

const isPromiseLike = (v: unknown): v is PromiseLike<unknown> =>
  v !== null && (typeof v === "object" || typeof v === "function") && typeof (v as { then?: unknown }).then === "function";

export class Tracer {
  readonly project: string;
  readonly processor: BatchProcessor | null;
  readonly sampleRate: number;
  readonly enabled: boolean;
  readonly limits: Limits;
  readonly redactor: Redactor;
  readonly captureArgs: boolean;
  #started = 0;
  #serializeErrors = 0;

  constructor(o: TracerOptions = {}) {
    const rate = o.sampleRate ?? 1;
    if (!(rate >= 0 && rate <= 1)) throw new RangeError("sampleRate must be in [0, 1]");
    this.project = o.project ?? "default";
    this.processor = o.processor ?? null;
    this.sampleRate = rate;
    this.enabled = o.enabled ?? true;
    this.limits = o.limits ?? DEFAULT_LIMITS;
    this.redactor = o.redactor ?? defaultRedactor();
    this.captureArgs = o.captureArgs ?? true;
  }

  /** Create a span without making it current (manual lifecycle: call `end()`). */
  startSpan(name: string, o: SpanOptions = {}): Span {
    if (!this.enabled) return NOOP_SPAN;
    const parent = o.parent === undefined ? (storage.getStore() ?? null) : o.parent;
    if (parent && !parent.sampled) return NOOP_SPAN;         // whole trace sampled out
    let traceId: string | undefined;
    if (!parent) {
      traceId = newTraceId();                                   // decide sampling on the trace id itself
      if (!sampleDecision(traceId, this.sampleRate)) return NOOP_SPAN;
    }
    this.#started += 1;
    const s = new Span({ name, type: o.type ?? "function", parent, sink: this, ...(traceId ? { traceId } : {}) });
    const init: LogFields = {};
    if (o.input !== undefined) init.input = o.input;
    if (o.expected !== undefined) init.expected = o.expected;
    if (o.metadata) init.metadata = o.metadata;
    if (o.tags) init.tags = o.tags;
    return s.log(init);
  }

  /**
   * Run `fn` inside a new current span. Works with sync and async functions: the span ends when
   * the function returns or its promise settles. Errors are recorded on the span and rethrown.
   */
  span<T>(name: string, fn: (span: Span) => T): T;
  span<T>(name: string, opts: SpanOptions, fn: (span: Span) => T): T;
  span<T>(name: string, a: SpanOptions | ((span: Span) => T), b?: (span: Span) => T): T {
    const [opts, fn] = typeof a === "function" ? [{}, a] : [a, b!];
    const s = this.startSpan(name, opts);
    if (s === NOOP_SPAN && !this.enabled) return fn(s);
    let out: T;
    try {
      out = storage.run(s, () => fn(s));
    } catch (e) {
      s.setError(e);
      s.end();
      throw e;
    }
    if (isPromiseLike(out)) {
      return Promise.resolve(out).then(
        (v) => { s.end(); return v; },
        (e: unknown) => { s.setError(e); s.end(); throw e; },
      ) as T;
    }
    s.end();
    return out;
  }

  /** Wrap a function so every call is a span. Sync, async, generator and async-generator functions. */
  traced<F extends (...args: never[]) => unknown>(fn: F, o: TracedOptions = {}): F {
    return wrap(() => this, fn, o, () => this.captureArgs);
  }

  current(): Span | undefined {
    return storage.getStore();
  }

  /** @internal SpanSink: serialize once at end, hand to the processor; never throws. */
  onEnd(span: Span): void {
    if (!this.processor) return;
    try {
      this.processor.onEnd(span.toRecord(this.project, this.limits, this.redactor));
    } catch {
      this.#serializeErrors += 1;
    }
  }

  flush(timeoutMs = 5000): Promise<boolean> {
    return this.processor ? this.processor.flush(timeoutMs) : Promise.resolve(true);
  }

  async shutdown(timeoutMs = 5000): Promise<void> {
    await this.processor?.shutdown(timeoutMs);
  }

  stats(): Record<string, unknown> {
    return { spans_started: this.#started, serialize_errors: this.#serializeErrors, ...(this.processor?.stats() ?? {}) };
  }
}

function wrap<F extends (...args: never[]) => unknown>(
  tracer: () => Tracer, fn: F, o: TracedOptions, defaultCapture: () => boolean,
): F {
  const name = o.name ?? (fn.name || "anonymous");
  const type = o.type ?? "function";
  const kind = fn.constructor?.name;
  const inputOf = (args: unknown[]): unknown => (args.length === 1 ? args[0] : args);
  const capArgs = () => o.captureArgs ?? defaultCapture();

  if (kind === "GeneratorFunction" || kind === "AsyncGeneratorFunction") {
    // The span is current only while the generator body runs, never while it's suspended at a
    // yield; otherwise it would leak into the consumer's context (same rule as the Python SDK).
    const isAsync = kind === "AsyncGeneratorFunction";
    const wrapped = function (this: unknown, ...args: unknown[]) {
      const t = tracer();
      const s = t.startSpan(name, capArgs() ? { type, input: inputOf(args) } : { type });
      const it = (fn as unknown as (...a: unknown[]) => Iterator<unknown> & AsyncIterator<unknown>).apply(this, args);
      let n = 0;
      let done = false;
      const finish = (e?: unknown) => {
        if (done) return;
        done = true;
        if (e !== undefined) s.setError(e);
        s.log({ metrics: { items: n } });
        s.end();
      };
      const step = (method: "next" | "return" | "throw", arg?: unknown) => {
        let r: unknown;
        try {
          r = storage.run(s, () => (it[method] as (x?: unknown) => unknown).call(it, arg));
        } catch (e) {
          finish(e);
          throw e;
        }
        const after = (res: IteratorResult<unknown>) => {
          if (res.done) finish();
          else n += 1;
          return res;
        };
        if (isAsync) {
          return (r as Promise<IteratorResult<unknown>>).then(after, (e: unknown) => { finish(e); throw e; });
        }
        return after(r as IteratorResult<unknown>);
      };
      const proxy = {
        next: (v?: unknown) => step("next", v),
        return: (v?: unknown) => step("return", v),
        throw: (e?: unknown) => step("throw", e),
        [isAsync ? Symbol.asyncIterator : Symbol.iterator]() { return proxy; },
      };
      return proxy;
    };
    Object.defineProperty(wrapped, "name", { value: fn.name });
    return wrapped as unknown as F;
  }

  const wrapped = function (this: unknown, ...args: unknown[]) {
    const t = tracer();
    return t.span(name, capArgs() ? { type, input: inputOf(args) } : { type }, (s) => {
      const out = (fn as unknown as (...a: unknown[]) => unknown).apply(this, args);
      if (o.captureOutput === false || !s.sampled) return out;
      if (isPromiseLike(out)) return Promise.resolve(out).then((v) => { s.log({ output: v }); return v; });
      s.log({ output: out });
      return out;
    });
  };
  Object.defineProperty(wrapped, "name", { value: fn.name });
  return wrapped as unknown as F;
}

// ---- process-global tracer -------------------------------------------------------------------
const state = globalState();
state.tracer ??= new Tracer({ enabled: false }); // no-op until init(): everything works, nothing is exported

export function getTracer(): Tracer {
  return state.tracer as Tracer;
}

/** Install `t` as the global tracer; the previous one is flushed and shut down in the background. */
export function setTracer(t: Tracer): Tracer {
  const old = state.tracer as Tracer;
  state.tracer = t;
  if (old !== t) void old.shutdown(1000).catch(() => undefined);
  return t;
}

export interface InitOptions extends BatchProcessorOptions {
  exporter?: Exporter;
  sampleRate?: number;
  redactEmails?: boolean;
  limits?: Limits;
  enabled?: boolean;
  captureArgs?: boolean;
}

function exporterFromEnv(project: string): Exporter {
  const spec = process.env.EVALWRIGHT_EXPORT ?? "";
  const i = spec.indexOf(":");
  const [kind, arg] = i < 0 ? [spec, ""] : [spec.slice(0, i), spec.slice(i + 1)];
  switch (kind) {
    case "otlp":
      return new OtlpHttpExporter(arg || "http://localhost:4318/v1/traces", project);
    case "http":
      if (!arg) throw new Error("EVALWRIGHT_EXPORT=http:<url> needs a URL");
      return new HttpExporter(arg, process.env.EVALWRIGHT_API_KEY ? { apiKey: process.env.EVALWRIGHT_API_KEY } : {});
    case "console":
      return new ConsoleExporter();
    case "memory":
      return new InMemoryExporter();
    default:
      return new JsonlExporter(arg || join(".evalwright", "traces", `${project}.jsonl`));
  }
}

/**
 * Install the global tracer. Environment (same as Python): EVALWRIGHT_EXPORT=jsonl:<path>|otlp:<url>|
 * http:<url>|console|memory, EVALWRIGHT_API_KEY, EVALWRIGHT_SAMPLE_RATE, EVALWRIGHT_DISABLED=1.
 */
export function init(project = "default", o: InitOptions = {}): Tracer {
  const enabled = o.enabled ?? !["1", "true", "yes"].includes((process.env.EVALWRIGHT_DISABLED ?? "").toLowerCase());
  const sampleRate = o.sampleRate ?? Number(process.env.EVALWRIGHT_SAMPLE_RATE ?? "1");
  const { exporter, redactEmails, limits, captureArgs, sampleRate: _s, enabled: _e, ...procOpts } = o;
  const exp = exporter ?? exporterFromEnv(project);
  return setTracer(new Tracer({
    project,
    processor: enabled ? new BatchProcessor(exp, procOpts) : null,
    sampleRate,
    redactor: defaultRedactor(redactEmails ?? false),
    ...(limits ? { limits } : {}),
    ...(captureArgs !== undefined ? { captureArgs } : {}),
    enabled,
  }));
}

export function span<T>(name: string, fn: (span: Span) => T): T;
export function span<T>(name: string, opts: SpanOptions, fn: (span: Span) => T): T;
export function span<T>(name: string, a: SpanOptions | ((span: Span) => T), b?: (span: Span) => T): T {
  return typeof a === "function" ? getTracer().span(name, a) : getTracer().span(name, a, b!);
}

/** Wrap a function; resolves the global tracer at call time (wrap at import, init later). */
export function traced<F extends (...args: never[]) => unknown>(fn: F, o: TracedOptions = {}): F {
  return wrap(getTracer, fn, o, () => getTracer().captureArgs);
}

export function currentSpan(): Span | undefined {
  return storage.getStore();
}

/** Log onto the current span (no-op outside a span). */
export function log(fields: LogFields): void {
  storage.getStore()?.log(fields);
}

export function flush(timeoutMs = 5000): Promise<boolean> {
  return getTracer().flush(timeoutMs);
}
