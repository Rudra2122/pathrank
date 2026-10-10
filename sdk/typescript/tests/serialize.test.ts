import { describe, expect, it } from "vitest";
import { defaultRedactor, toJsonable } from "../src/index.js";

const LIM = { maxDepth: 3, maxString: 50, maxItems: 3 };

describe("toJsonable is total and bounded", () => {
  it("handles every primitive and odd value without throwing", () => {
    const sym = Symbol("s");
    const out = toJsonable({
      u: undefined, n: null, b: true, nan: NaN, inf: -Infinity, big: 12n, huge: 2n ** 70n, s: "x",
      sym, fn: function named() {}, arrow: () => 1, date: new Date(0), bad: new Date("nope"),
      err: new TypeError("boom"), re: /a+/g, bytes: new Uint8Array(5), ab: new ArrayBuffer(3),
      f32: new Float32Array([1, 2]), map: new Map([["k", 1]]), set: new Set([1, 2]),
    });
    expect(out).toMatchObject({
      u: null, n: null, b: true, nan: "NaN", inf: "-Infinity", big: 12, huge: "1180591620717411303424", s: "x",
      sym: "Symbol(s)", fn: "<function named>", date: "1970-01-01T00:00:00.000Z", bad: "Invalid Date",
      err: { type: "TypeError", message: "boom" }, re: "/a+/g", bytes: "<bytes len=5>", ab: "<bytes len=3>",
      f32: [1, 2], map: { k: 1 }, set: [1, 2],
    });
  });

  it("cuts cycles, depth, items and long strings", () => {
    const a: Record<string, unknown> = { name: "a" };
    a.self = a;
    expect(toJsonable(a)).toEqual({ name: "a", self: "<cycle: Object>" });
    expect(toJsonable({ a: { b: { c: { d: 1 } } } }, LIM)).toEqual({ a: { b: { c: "<max depth: Object>" } } });
    expect(toJsonable([1, 2, 3, 4, 5], LIM)).toEqual([1, 2, 3, "...2 more items"]);
    expect(toJsonable({ a: 1, b: 2, c: 3, d: 4 }, LIM)).toEqual({ a: 1, b: 2, c: 3, "...": "1 more keys" });
    expect((toJsonable("y".repeat(80), LIM) as string).startsWith("y".repeat(50) + "...[truncated 30 chars]")).toBe(true);
  });

  it("shared (non-cyclic) references are kept, not reported as cycles", () => {
    const shared = { x: 1 };
    expect(toJsonable({ a: shared, b: shared })).toEqual({ a: { x: 1 }, b: { x: 1 } });
  });

  it("class instances get a __type__ and toJSON is respected", () => {
    class Doc { constructor(public id = 7) {} }
    expect(toJsonable(new Doc())).toEqual({ __type__: "Doc", id: 7 });
    expect(toJsonable({ toJSON: () => ({ ok: true }) })).toEqual({ ok: true });
  });

  it("a throwing getter can't escape", () => {
    const evil = { get boom(): never { throw new Error("nope"); } };
    expect(toJsonable(evil)).toMatch(/^<unserializable object/);
  });

  it("emails are redacted only when asked", () => {
    expect(toJsonable("mail a@b.co", undefined, defaultRedactor(true))).toBe("mail [EMAIL]");
    expect(toJsonable("mail a@b.co", undefined, defaultRedactor(false))).toBe("mail a@b.co");
  });
});
