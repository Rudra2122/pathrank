import { appendFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { gzipSync } from "node:zlib";
import { batchKey } from "./ids.js";
import type { SpanRecord } from "./span.js";
import { VERSION } from "./version.js";

export interface ExportResult {
  success: boolean;
  retryable: boolean;
  retryAfterMs?: number;
  error?: string;
}

export const ExportResult = {
  ok: (): ExportResult => ({ success: true, retryable: false }),
  retry: (error: string, retryAfterMs?: number): ExportResult =>
    retryAfterMs === undefined ? { success: false, retryable: true, error } : { success: false, retryable: true, error, retryAfterMs },
  fail: (error: string): ExportResult => ({ success: false, retryable: false, error }),
};

/** Anything that can ship a batch. May throw; a throw is treated as retryable. */
export interface Exporter {
  export(batch: SpanRecord[]): Promise<ExportResult>;
  shutdown?(): Promise<void>;
}

export class InMemoryExporter implements Exporter {
  readonly spans: SpanRecord[] = [];
  async export(batch: SpanRecord[]): Promise<ExportResult> {
    this.spans.push(...batch);
    return ExportResult.ok();
  }
}

export class ConsoleExporter implements Exporter {
  async export(batch: SpanRecord[]): Promise<ExportResult> {
    for (const s of batch) {
      const err = s.status === "error" ? ` ERROR ${s.error?.type}: ${s.error?.message}` : "";
      process.stderr.write(`[evalwright] ${s.type.padEnd(8)} ${s.name} ${s.duration_ms.toFixed(1)}ms${err}\n`);
    }
    return ExportResult.ok();
  }
}

export class JsonlExporter implements Exporter {
  #ready: Promise<unknown> | null = null;
  constructor(readonly path: string) {}
  async export(batch: SpanRecord[]): Promise<ExportResult> {
    this.#ready ??= mkdir(dirname(this.path), { recursive: true });
    await this.#ready;
    await appendFile(this.path, batch.map((s) => JSON.stringify(s)).join("\n") + "\n", "utf8");
    return ExportResult.ok();
  }
}

const RETRYABLE = new Set([408, 425, 429, 500, 502, 503, 504]);

export function parseRetryAfter(v: string | null): number | undefined {
  if (!v) return undefined;
  const secs = Number(v);
  if (Number.isFinite(secs)) return Math.max(0, secs * 1000);
  const at = Date.parse(v);
  return Number.isNaN(at) ? undefined : Math.max(0, at - Date.now());
}

export interface HttpExporterOptions {
  apiKey?: string;
  headers?: Record<string, string>;
  timeoutMs?: number;
  gzip?: boolean;
  fetch?: typeof fetch;
}

/**
 * POST batches to an ingestion endpoint (wire protocol v1).
 * 2xx ok; 408/425/429/5xx/network/timeout retryable (Retry-After honored); other 4xx dropped.
 */
export class HttpExporter implements Exporter {
  readonly url: string;
  readonly #headers: Record<string, string>;
  readonly #timeoutMs: number;
  readonly #gzip: boolean;
  readonly #fetch: typeof fetch;

  constructor(url: string, opts: HttpExporterOptions = {}) {
    this.url = url;
    this.#timeoutMs = opts.timeoutMs ?? 10_000;
    this.#gzip = opts.gzip ?? true;
    this.#fetch = opts.fetch ?? globalThis.fetch;
    this.#headers = { "Content-Type": "application/json", "User-Agent": userAgent(), ...opts.headers };
    if (opts.apiKey) this.#headers.Authorization = `Bearer ${opts.apiKey}`;
  }

  payload(batch: SpanRecord[]): unknown {
    return { spans: batch };
  }

  async export(batch: SpanRecord[]): Promise<ExportResult> {
    const json = JSON.stringify(this.payload(batch));
    const headers: Record<string, string> = { ...this.#headers, "Idempotency-Key": batchKey(batch) };
    let body: string | Uint8Array = json;
    if (this.#gzip) {
      body = gzipSync(json);
      headers["Content-Encoding"] = "gzip";
    }
    try {
      const res = await this.#fetch(this.url, {
        method: "POST",
        headers,
        body,
        signal: AbortSignal.timeout(this.#timeoutMs),
      });
      await res.arrayBuffer().catch(() => undefined); // drain so the socket can be reused
      if (res.ok) return ExportResult.ok();
      if (RETRYABLE.has(res.status) || res.status >= 500) {
        return ExportResult.retry(`HTTP ${res.status}`, parseRetryAfter(res.headers.get("retry-after")));
      }
      return ExportResult.fail(`HTTP ${res.status}`);
    } catch (e) {
      return ExportResult.retry(`${(e as Error)?.name ?? "Error"}: ${(e as Error)?.message ?? e}`);
    }
  }
}

/** OpenTelemetry OTLP/HTTP JSON, so spans land in any OTel collector (Jaeger, Tempo, Honeycomb...). */
export class OtlpHttpExporter extends HttpExporter {
  constructor(url = "http://localhost:4318/v1/traces", readonly serviceName = "evalwright", opts: HttpExporterOptions = {}) {
    super(url, { gzip: false, ...opts });
  }
  override payload(batch: SpanRecord[]): unknown {
    const attr = (key: string, value: unknown) => ({
      key,
      value: typeof value === "string" ? { stringValue: value } : typeof value === "number" ? { doubleValue: value } : { stringValue: JSON.stringify(value) },
    });
    const spans = batch.map((s) => {
      const attributes = [attr("evalwright.type", s.type), attr("evalwright.project", s.project)];
      for (const k of ["input", "output", "expected"] as const) if (s[k] !== null) attributes.push(attr(`evalwright.${k}`, s[k]));
      for (const [k, v] of Object.entries(s.metrics)) attributes.push(attr(`evalwright.metrics.${k}`, v));
      for (const [k, v] of Object.entries(s.scores)) if (v !== null) attributes.push(attr(`evalwright.scores.${k}`, v));
      if (Object.keys(s.metadata).length) attributes.push(attr("evalwright.metadata", s.metadata));
      return {
        traceId: s.trace_id,
        spanId: s.span_id,
        ...(s.parent_id ? { parentSpanId: s.parent_id } : {}),
        name: s.name,
        kind: 1,
        startTimeUnixNano: String(s.start_ns),
        endTimeUnixNano: String(s.end_ns),
        attributes,
        status: s.status === "error" ? { code: 2, message: s.error?.message ?? "" } : { code: 1 },
      };
    });
    return {
      resourceSpans: [{
        resource: { attributes: [attr("service.name", this.serviceName)] },
        scopeSpans: [{ scope: { name: "evalwright", version: VERSION }, spans }],
      }],
    };
  }
}

export function userAgent(): string {
  return `evalwright-typescript/${VERSION} node/${process.versions.node}`;
}
