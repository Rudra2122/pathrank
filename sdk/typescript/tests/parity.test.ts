/**
 * Cross-language parity: these expected values were produced by the Python SDK
 * (zlib.crc32, evalwright.export.exporters.batch_key, EvalCase.case_id, default redactor).
 * If either SDK changes one of these rules, this test and its Python counterpart must change together.
 */
import { describe, expect, it } from "vitest";
import { batchKey, caseId, crc32, defaultRedactor, toJsonable, VERSION } from "../src/index.js";
import pkg from "../package.json" with { type: "json" };

describe("parity with the Python SDK", () => {
  it("crc32 sampling hash matches zlib.crc32", () => {
    expect(crc32("0123456789abcdef0123456789abcdef")).toBe(2002367758);
    expect(crc32("ffffffffffffffffffffffffffffffff")).toBe(1885562385);
    expect(crc32("a3f1c2d4e5b60718293a4b5c6d7e8f90")).toBe(1003086553);
  });

  it("idempotency key matches batch_key", () => {
    expect(batchKey([{ span_id: "0123456789abcdef" }, { span_id: "fedcba9876543210" }])).toBe("f46549101424db271de4288330657c77");
  });

  it("eval case ids match EvalCase.case_id (so experiments compare across languages)", () => {
    const j = (x: unknown) => toJsonable(x);
    expect(caseId(j("hello"), j("world"))).toBe("9e0037a531c8e873");
    expect(caseId(j({ q: "what is 2+2", ctx: [1, 2, 3] }), j(4))).toBe("85c72cac33fdfe87");
    expect(caseId(j("héllo ünïcode"), null)).toBe("6ff37799aef168e9");
    expect(caseId(j([1, "a", true, null]), j({ b: [false] }))).toBe("0a77cc275b8f660c");
  });

  it("redaction matches the default redactor", () => {
    const out = toJsonable(
      { api_key: "abc", note: "Bearer abcdefghijklmnopqrstu and sk-ABCDEFGHIJKLMNOPQRST", password: "", n: 3 },
      undefined, defaultRedactor(),
    );
    expect(out).toEqual({ api_key: "[REDACTED]", note: "[REDACTED] and [REDACTED]", password: "", n: 3 });
  });

  it("truncation marker matches", () => {
    expect(toJsonable("x".repeat(20), { maxDepth: 8, maxString: 5, maxItems: 200 })).toBe("xxxxx...[truncated 15 chars]");
  });

  it("VERSION matches package.json", () => {
    expect(VERSION).toBe(pkg.version);
  });
});
