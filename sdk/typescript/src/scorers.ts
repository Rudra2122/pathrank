/**
 * Built-in scorers. Each returns a Scorer: a function of {input, output, expected, metadata}
 * that returns a score in [0, 1] (or null = not applicable). Names and semantics match the
 * Python SDK, so scores from both languages compare directly.
 */
import type { Score, Scorer, ScorerArgs } from "./evals.js";

/** Strings as-is; everything else as JSON with sorted keys (stable comparisons). */
export function text(x: unknown): string {
  if (typeof x === "string") return x;
  return JSON.stringify(sortKeys(x)) ?? String(x);
}

function sortKeys(x: unknown): unknown {
  if (Array.isArray(x)) return x.map(sortKeys);
  if (x && typeof x === "object" && Object.getPrototypeOf(x) === Object.prototype) {
    return Object.fromEntries(Object.keys(x).sort().map((k) => [k, sortKeys((x as Record<string, unknown>)[k])]));
  }
  return x;
}

function named<S extends Scorer>(name: string, fn: S): S {
  fn.scorerName = name;
  return fn;
}

export function exactMatch(o: { caseSensitive?: boolean; trim?: boolean; name?: string } = {}): Scorer {
  return named(o.name ?? "exact_match", ({ output, expected }: ScorerArgs) => {
    if (expected === undefined || expected === null) return null;
    let a = text(output);
    let b = text(expected);
    if (o.trim ?? true) [a, b] = [a.trim(), b.trim()];
    if (!(o.caseSensitive ?? true)) [a, b] = [a.toLowerCase(), b.toLowerCase()];
    return a === b ? 1 : 0;
  });
}

export function contains(o: { mode?: "all" | "any" | "fraction"; caseSensitive?: boolean; name?: string } = {}): Scorer {
  const mode = o.mode ?? "all";
  const cs = o.caseSensitive ?? false;
  return named(o.name ?? "contains", ({ output, expected }: ScorerArgs) => {
    if (expected === undefined || expected === null) return null;
    const needles = Array.isArray(expected) ? expected : [expected];
    if (!needles.length) return null;
    const hay = cs ? text(output) : text(output).toLowerCase();
    const hits = needles.map((n) => hay.includes(cs ? String(n) : String(n).toLowerCase()));
    if (mode === "all") return hits.every(Boolean) ? 1 : 0;
    if (mode === "any") return hits.some(Boolean) ? 1 : 0;
    return hits.filter(Boolean).length / hits.length;
  });
}

export function jsonValid(o: { requiredKeys?: string[]; name?: string } = {}): Scorer {
  return named(o.name ?? "json_valid", ({ output }: ScorerArgs) => {
    let obj: unknown = output;
    if (typeof output === "string") {
      try {
        obj = JSON.parse(output);
      } catch {
        return 0;
      }
    }
    if (o.requiredKeys?.length) {
      return obj !== null && typeof obj === "object" && !Array.isArray(obj) && o.requiredKeys.every((k) => k in (obj as object)) ? 1 : 0;
    }
    return obj !== null && typeof obj === "object" ? 1 : 0;
  });
}

export function levenshteinDistance(a: string, b: string): number {
  if (a.length < b.length) [a, b] = [b, a];
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur.push(Math.min(prev[j]! + 1, cur[j - 1]! + 1, prev[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1)));
    }
    prev = cur;
  }
  return prev[b.length]!;
}

/** 1 - edit distance / longer length. Long strings are truncated to `maxLen` (O(n*m) cost). */
export function levenshtein(o: { maxLen?: number; name?: string } = {}): Scorer {
  const maxLen = o.maxLen ?? 2000;
  return named(o.name ?? "levenshtein", ({ output, expected }: ScorerArgs) => {
    if (expected === undefined || expected === null) return null;
    const a = text(output).slice(0, maxLen);
    const b = text(expected).slice(0, maxLen);
    if (!a && !b) return 1;
    return 1 - levenshteinDistance(a, b) / Math.max(a.length, b.length);
  });
}

export function numericCloseness(o: { eps?: number; name?: string } = {}): Scorer {
  const eps = o.eps ?? 1e-9;
  return named(o.name ?? "numeric_closeness", ({ output, expected }: ScorerArgs) => {
    if (expected === undefined || expected === null) return null;
    const x = Number(output);
    const e = Number(expected);
    if (!Number.isFinite(x) || !Number.isFinite(e)) return 0;
    return Math.max(0, 1 - Math.abs(x - e) / Math.max(Math.abs(e), eps));
  });
}

export function regexMatch(pattern: RegExp | string, o: { name?: string } = {}): Scorer {
  const rx = typeof pattern === "string" ? new RegExp(pattern) : pattern;
  return named(o.name ?? "regex_match", ({ output }: ScorerArgs) => {
    rx.lastIndex = 0;
    return rx.test(text(output)) ? 1 : 0;
  });
}

const JUDGE_TEMPLATE = `You are grading the output of an AI system.

Criteria: {rubric}

Input:
{input}

Expected (may be empty):
{expected}

Output:
{output}

Reply with a short rationale, then on the final line write exactly "Choice: X" where X is one of: {choices}.
`;

/**
 * Model-agnostic LLM-as-judge: `llm` is any (prompt) => text function, so it works with any
 * provider's client. The final "Choice: X" line maps to a score via `choices`.
 */
export function llmJudge(
  llm: (prompt: string) => string | Promise<string>,
  o: { rubric: string; choices?: Record<string, number>; name?: string; template?: string },
): Scorer {
  const choices = o.choices ?? { A: 1, B: 0.5, C: 0 };
  const name = o.name ?? "llm_judge";
  return named(name, async ({ input, output, expected }: ScorerArgs): Promise<Score> => {
    const prompt = (o.template ?? JUDGE_TEMPLATE)
      .replace("{rubric}", o.rubric)
      .replace("{input}", text(input))
      .replace("{expected}", expected === undefined || expected === null ? "" : text(expected))
      .replace("{output}", text(output))
      .replace("{choices}", Object.keys(choices).join(", "));
    const reply = String(await llm(prompt));
    const all = [...reply.matchAll(/Choice:\s*([A-Za-z0-9_]+)/g)];
    const choice = all.length ? all[all.length - 1]![1]! : null;
    if (choice === null || !(choice in choices)) {
      return { name, score: null, metadata: { error: "unparseable judge reply", reply: reply.slice(-500) } };
    }
    return { name, score: choices[choice]!, metadata: { choice, rationale: reply.slice(-1000) } };
  });
}
