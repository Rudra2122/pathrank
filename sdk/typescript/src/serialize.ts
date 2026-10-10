/**
 * toJsonable: make any value safe to send. Total (never throws), bounded (depth, string length,
 * item counts), cycle-safe, and redacting by default. Same rules as the Python SDK.
 */

export interface Limits {
  maxDepth: number;
  maxString: number;
  maxItems: number;
}

export const DEFAULT_LIMITS: Readonly<Limits> = Object.freeze({ maxDepth: 8, maxString: 10_000, maxItems: 200 });

export type Redactor = (key: string, value: unknown) => unknown;
export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };

const SECRET_KEY = /(pass(word)?|secret|token|api[_-]?key|authorization|auth|cookie|session|private[_-]?key)/i;
const SECRET_VALUE = /(sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{16,}|AKIA[0-9A-Z]{16})/g;
const EMAIL = /[\w.+-]+@[\w-]+\.[\w.]+/g;
export const REDACTED = "[REDACTED]";

const keyCache = new Map<string, boolean>();
function isSecretKey(key: string): boolean {
  let hit = keyCache.get(key);
  if (hit === undefined) {
    hit = SECRET_KEY.test(key);
    if (keyCache.size < 8192) keyCache.set(key, hit);
  }
  return hit;
}

/** Mask values under secret-looking keys, and secret-looking strings anywhere. */
export function defaultRedactor(redactEmails = false): Redactor {
  return (key, value) => {
    if (key && isSecretKey(key) && value !== null && value !== undefined && value !== "" && value !== 0 && value !== false) {
      return REDACTED;
    }
    if (typeof value === "string") {
      let v = value.replace(SECRET_VALUE, REDACTED);
      if (redactEmails) v = v.replace(EMAIL, "[EMAIL]");
      return v;
    }
    return value;
  };
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n)}...[truncated ${s.length - n} chars]`;
}

export function toJsonable(value: unknown, limits: Limits = DEFAULT_LIMITS, redactor?: Redactor): Json {
  try {
    return convert(value, limits, redactor, "", 0, new Set());
  } catch (e) {
    const t = value === null ? "null" : typeof value;
    return `<unserializable ${t}: ${(e as Error)?.name ?? "Error"}>`;
  }
}

function convert(v: unknown, lim: Limits, red: Redactor | undefined, key: string, depth: number, seen: Set<object>): Json {
  if (red) v = red(key, v);
  switch (typeof v) {
    case "undefined":
      return null;
    case "boolean":
      return v;
    case "number":
      return Number.isFinite(v) ? v : String(v);
    case "bigint":
      return v >= -(2n ** 63n) && v < 2n ** 63n && Number.isSafeInteger(Number(v)) ? Number(v) : v.toString();
    case "string":
      return truncate(v, lim.maxString);
    case "symbol":
      return v.toString();
    case "function":
      return `<function ${v.name || "anonymous"}>`;
  }
  if (v === null) return null;
  const obj = v as object;
  if (obj instanceof Date) return Number.isNaN(obj.getTime()) ? "Invalid Date" : obj.toISOString();
  if (obj instanceof Error) return { type: obj.name, message: truncate(obj.message, lim.maxString) };
  if (obj instanceof ArrayBuffer) return `<bytes len=${obj.byteLength}>`;
  if (ArrayBuffer.isView(obj)) {
    if (obj instanceof Uint8Array || obj instanceof DataView) return `<bytes len=${obj.byteLength}>`;
    const arr = Array.from(obj as unknown as ArrayLike<number | bigint>);
    return convert(arr, lim, undefined, key, depth, seen);
  }
  if (obj instanceof RegExp) return obj.toString();
  if (depth >= lim.maxDepth) return `<max depth: ${obj.constructor?.name ?? "Object"}>`;
  if (seen.has(obj)) return `<cycle: ${obj.constructor?.name ?? "Object"}>`;
  seen.add(obj);
  try {
    const next = depth + 1;
    if (Array.isArray(obj) || obj instanceof Set) {
      const items = Array.isArray(obj) ? obj : Array.from(obj);
      const out: Json[] = items.slice(0, lim.maxItems).map((x) => convert(x, lim, red, key, next, seen));
      if (items.length > lim.maxItems) out.push(`...${items.length - lim.maxItems} more items`);
      return out;
    }
    if (obj instanceof Map) {
      const out: { [k: string]: Json } = {};
      let i = 0;
      for (const [k, val] of obj) {
        if (i++ >= lim.maxItems) {
          out["..."] = `${obj.size - lim.maxItems} more keys`;
          break;
        }
        const sk = typeof k === "string" ? k : String(k);
        out[sk] = convert(val, lim, red, sk, next, seen);
      }
      return out;
    }
    const toJSON = (obj as { toJSON?: unknown }).toJSON;
    if (typeof toJSON === "function") return convert(toJSON.call(obj), lim, red, key, next, seen);
    const proto = Object.getPrototypeOf(obj);
    const out: { [k: string]: Json } = {};
    if (proto !== null && proto !== Object.prototype) out.__type__ = proto.constructor?.name ?? "Object";
    const keys = Object.keys(obj);
    for (let i = 0; i < keys.length; i++) {
      if (i >= lim.maxItems) {
        out["..."] = `${keys.length - lim.maxItems} more keys`;
        break;
      }
      const k = keys[i]!;
      out[k] = convert((obj as Record<string, unknown>)[k], lim, red, k, next, seen);
    }
    return out;
  } finally {
    seen.delete(obj);
  }
}
