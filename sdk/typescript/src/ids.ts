import { createHash, randomBytes } from "node:crypto";

/** 32 lowercase hex chars (128 bits), same shape as the Python SDK. */
export function newTraceId(): string {
  return randomBytes(16).toString("hex");
}

/** 16 lowercase hex chars (64 bits). */
export function newSpanId(): string {
  return randomBytes(8).toString("hex");
}

const CRC_TABLE: Uint32Array = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** CRC-32 (IEEE), bit-identical to Python's zlib.crc32 for ASCII input. */
export function crc32(s: string): number {
  let c = 0xffffffff;
  for (let i = 0; i < s.length; i++) c = CRC_TABLE[(c ^ s.charCodeAt(i)) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/**
 * Deterministic head sampling: the same trace id gets the same decision in every SDK,
 * so a trace that crosses a Python service and a Node service is kept or dropped as a whole.
 */
export function sampled(traceId: string, rate: number): boolean {
  if (rate >= 1) return true;
  if (rate <= 0) return false;
  return crc32(traceId) / 0xffffffff < rate;
}

/** Wire protocol v1 idempotency key: sha256 of span ids joined by "\n", first 32 hex chars. */
export function batchKey(batch: ReadonlyArray<{ span_id?: unknown }>): string {
  return createHash("sha256")
    .update(batch.map((r) => String(r.span_id ?? "")).join("\n"))
    .digest("hex")
    .slice(0, 32);
}

/** Stable eval case id: sha1 of compact JSON {input, expected}, first 16 hex chars (matches Python). */
export function caseId(input: unknown, expected: unknown): string {
  return createHash("sha1").update(JSON.stringify({ input, expected: expected ?? null })).digest("hex").slice(0, 16);
}
