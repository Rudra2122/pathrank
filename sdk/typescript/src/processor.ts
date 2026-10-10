import type { Exporter, ExportResult } from "./exporters.js";
import type { SpanRecord } from "./span.js";

export interface RetryPolicyOptions {
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
}

export class RetryPolicy {
  readonly maxRetries: number;
  readonly baseDelayMs: number;
  readonly maxDelayMs: number;
  constructor(o: RetryPolicyOptions = {}) {
    this.maxRetries = o.maxRetries ?? 5;
    this.baseDelayMs = o.baseDelayMs ?? 500;
    this.maxDelayMs = o.maxDelayMs ?? 30_000;
  }
  /** Exponential backoff with full jitter; a server's Retry-After wins (capped). */
  delayMs(attempt: number, retryAfterMs: number | undefined, random: () => number): number {
    if (retryAfterMs !== undefined) return Math.min(Math.max(retryAfterMs, 0), this.maxDelayMs);
    return random() * Math.min(this.maxDelayMs, this.baseDelayMs * 2 ** attempt);
  }
}

export type CircuitState = "closed" | "open" | "half_open";

/** Opens after `threshold` consecutive failures; after `cooldownMs` lets one probe through. */
export class CircuitBreaker {
  failures = 0;
  openedAt: number | null = null;
  constructor(readonly threshold = 5, readonly cooldownMs = 30_000, private readonly clock: () => number = () => performance.now()) {}
  get state(): CircuitState {
    if (this.openedAt === null) return "closed";
    return this.clock() - this.openedAt >= this.cooldownMs ? "half_open" : "open";
  }
  allow(): boolean {
    return this.state !== "open";
  }
  record(ok: boolean): void {
    if (ok) {
      this.failures = 0;
      this.openedAt = null;
      return;
    }
    this.failures += 1;
    if (this.failures >= this.threshold || this.state === "half_open") this.openedAt = this.clock();
  }
}

export interface BatchProcessorOptions {
  maxQueue?: number;
  maxBatch?: number;
  flushIntervalMs?: number;
  retry?: RetryPolicy;
  breaker?: CircuitBreaker;
  random?: () => number;
  /** Flush on process exit (default true). */
  exitHook?: boolean;
}

const unrefSleep = (ms: number): Promise<void> =>
  new Promise((r) => {
    const t = setTimeout(r, ms);
    t.unref?.();
  });

/**
 * Background exporter. Guarantees (wire protocol v1):
 * never throws into the caller, never blocks it (enqueue is O(1) amortized), bounded memory
 * (drop-oldest), retries transient failures, pauses a failing endpoint, flushes on exit.
 * Timers are unref'd: the processor never keeps a finished program alive by itself.
 */
export class BatchProcessor {
  readonly exporter: Exporter;
  readonly retry: RetryPolicy;
  readonly breaker: CircuitBreaker;
  readonly #maxQueue: number;
  readonly #maxBatch: number;
  readonly #random: () => number;
  #queue: SpanRecord[] = [];
  #draining: Promise<void> | null = null;
  #inflight = 0;
  #timer: ReturnType<typeof setInterval> | null = null;
  #closed = false;
  #lastError: string | null = null;
  #exitHook: (() => void) | null = null;
  readonly counters = {
    enqueued: 0, exported: 0, dropped_queue_full: 0, dropped_failed: 0, batches_failed: 0, export_calls: 0, retries: 0,
  };

  constructor(exporter: Exporter, o: BatchProcessorOptions = {}) {
    if ((o.maxBatch ?? 1) <= 0 || (o.maxQueue ?? 1) <= 0) throw new RangeError("maxBatch and maxQueue must be positive");
    this.exporter = exporter;
    this.#maxQueue = o.maxQueue ?? 10_000;
    this.#maxBatch = o.maxBatch ?? 512;
    this.retry = o.retry ?? new RetryPolicy();
    this.breaker = o.breaker ?? new CircuitBreaker();
    this.#random = o.random ?? Math.random;
    this.#timer = setInterval(() => this.#kick(), o.flushIntervalMs ?? 1000);
    this.#timer.unref?.();
    if (o.exitHook ?? true) {
      let busy = false;
      this.#exitHook = () => {
        if (busy || this.#queue.length === 0) return;   // beforeExit can fire repeatedly
        busy = true;
        void this.flush(2000).finally(() => { busy = false; });
      };
      process.on("beforeExit", this.#exitHook);
    }
  }

  onEnd(record: SpanRecord): void {
    if (this.#closed) return;
    if (this.#queue.length >= this.#maxQueue) {
      this.#queue.shift();
      this.counters.dropped_queue_full += 1;
    }
    this.#queue.push(record);
    this.counters.enqueued += 1;
    if (this.#queue.length >= this.#maxBatch) queueMicrotask(() => this.#kick());
  }

  #kick(): Promise<void> {
    if (!this.#draining && this.#queue.length > 0) {
      this.#draining = this.#drain().finally(() => { this.#draining = null; });
    }
    return this.#draining ?? Promise.resolve();
  }

  async #drain(): Promise<void> {
    while (this.#queue.length > 0 && this.breaker.allow()) {
      const batch = this.#queue.splice(0, this.#maxBatch);
      this.#inflight = batch.length;
      try {
        const ok = await this.#exportWithRetry(batch);
        if (ok) this.counters.exported += batch.length;
        else {
          this.counters.dropped_failed += batch.length;
          this.counters.batches_failed += 1;
        }
      } finally {
        this.#inflight = 0;
      }
    }
  }

  async #exportWithRetry(batch: SpanRecord[]): Promise<boolean> {
    for (let attempt = 0; ; attempt++) {
      this.counters.export_calls += 1;
      let res: ExportResult;
      try {
        res = await this.exporter.export(batch);
      } catch (e) {
        res = { success: false, retryable: true, error: `${(e as Error)?.name}: ${(e as Error)?.message}` };
      }
      if (res.success) {
        this.breaker.record(true);
        return true;
      }
      this.#lastError = res.error ?? "unknown";
      this.breaker.record(false);
      if (!res.retryable || attempt >= this.retry.maxRetries || !this.breaker.allow()) return false;
      this.counters.retries += 1;
      await unrefSleep(this.retry.delayMs(attempt, res.retryAfterMs, this.#random));
    }
  }

  /** Resolve true when everything queued so far was delivered, false on timeout or drops. */
  async flush(timeoutMs = 5000): Promise<boolean> {
    const deadline = performance.now() + timeoutMs;
    const keepAlive = setTimeout(() => undefined, timeoutMs);   // ref'd: keep the process up while flushing
    try {
      const failedBefore = this.counters.dropped_failed;
      while (this.#queue.length > 0 || this.#draining) {
        const left = deadline - performance.now();
        if (left <= 0) return false;
        if (!this.breaker.allow() && !this.#draining) return false;
        await Promise.race([this.#kick(), unrefSleep(Math.min(left, 50))]);
      }
      return this.counters.dropped_failed === failedBefore;
    } finally {
      clearTimeout(keepAlive);
    }
  }

  async shutdown(timeoutMs = 5000): Promise<void> {
    await this.flush(timeoutMs);
    this.#closed = true;
    if (this.#timer) clearInterval(this.#timer);
    if (this.#exitHook) process.off("beforeExit", this.#exitHook);
    await this.exporter.shutdown?.();
  }

  stats(): Record<string, unknown> {
    return { ...this.counters, queued: this.#queue.length, inflight: this.#inflight, circuit: this.breaker.state, last_error: this.#lastError };
  }
}
