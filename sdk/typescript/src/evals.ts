import { mkdir, readdir, readFile, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { globalState } from "./global.js";
import { caseId } from "./ids.js";
import { type Json, toJsonable } from "./serialize.js";
import type { Span } from "./span.js";
import { bootstrapCi, mean, percentile, stderr } from "./stats.js";
import { getTracer, type Tracer } from "./tracer.js";

// ---- types ------------------------------------------------------------------------------------
export interface EvalCase<I = unknown, E = unknown> {
  input: I;
  expected?: E;
  metadata?: Record<string, unknown>;
  id?: string;
}

export interface Score {
  name: string;
  score: number | null;
  metadata?: Record<string, unknown>;
}

export interface ScorerArgs<I = unknown, O = unknown, E = unknown> {
  input: I;
  output: O;
  expected: E | undefined;
  metadata: Record<string, unknown>;
}

export type ScorerResult = number | boolean | null | undefined | Score | Score[] | Record<string, number | null>;

/** A scorer is a (possibly async) function. Its `name` (or `scorerName`) names the score. */
export type Scorer<I = unknown, O = unknown, E = unknown> =
  ((args: ScorerArgs<I, O, E>) => ScorerResult | Promise<ScorerResult>) & { scorerName?: string };

export interface TaskHooks {
  metadata: Record<string, unknown>;
  expected: unknown;
  span: Span;
}

export interface EvalOptions<I, O, E> {
  data: Iterable<EvalCase<I, E> | I> | (() => Iterable<EvalCase<I, E> | I> | Promise<Iterable<EvalCase<I, E> | I>>);
  task: (input: I, hooks: TaskHooks) => O | Promise<O>;
  scores?: Scorer<I, O, E>[];
  maxConcurrency?: number;
  trials?: number;
  timeoutMs?: number;
  /** Where experiments are stored (same layout as the Python SDK); null disables storage. */
  storeDir?: string | null;
  /** Experiment to compare with: "latest" (default), "previous", an id, or null. */
  baseline?: string | null;
  metadata?: Record<string, unknown>;
  tracer?: Tracer;
  seed?: number;
}

export interface CaseRecord {
  case_id: string;
  trial: number;
  input: Json;
  expected: Json;
  metadata: Json;
  output: Json;
  scores: Record<string, number | null>;
  score_metadata: Record<string, Json>;
  error: { type: string; message: string } | null;
  scorer_errors?: Record<string, string>;
  duration_ms: number;
  trace_id: string;
}

export interface ScoreSummary {
  mean: number | null;
  stderr: number | null;
  ci95: [number, number] | null;
  n: number;
  min: number | null;
  max: number | null;
}

export interface EvalSummary {
  n_cases: number;
  trials: number;
  n_runs: number;
  errors: number;
  error_rate: number;
  scorer_errors: number;
  scores: Record<string, ScoreSummary>;
  latency_ms: { p50: number | null; p95: number | null };
  wall_seconds: number;
  metadata?: Json;
}

export interface ScoreComparison {
  baseline: number | null;
  current: number | null;
  diff: number | null;
  ci95: [number, number] | null;
  improved: number;
  regressed: number;
  unchanged: number;
  n: number;
}

export interface Comparison {
  baseline: string;
  current: string;
  common_cases: number;
  new_cases: number;
  missing_cases: number;
  scores: Record<string, ScoreComparison>;
}

export interface GateOptions {
  min?: Record<string, number>;
  maxRegression?: number;
  maxErrorRate?: number;
}

export class EvalResult {
  gateFailures: string[] = [];
  constructor(
    readonly name: string,
    readonly experimentId: string | null,
    readonly cases: CaseRecord[],
    readonly summary: EvalSummary,
    readonly comparison: Comparison | null,
    readonly path: string | null,
  ) {}

  score(name: string): number | null {
    return this.summary.scores[name]?.mean ?? null;
  }

  /** Gate failures (empty = pass): minimum scores, regression vs baseline, error rate. */
  check(g: GateOptions = {}): string[] {
    const fails: string[] = [];
    for (const [k, thr] of Object.entries(g.min ?? {})) {
      const v = this.score(k);
      if (v === null || v < thr) fails.push(`${this.name}: ${k}=${v} below minimum ${thr}`);
    }
    if (g.maxRegression !== undefined && this.comparison) {
      for (const [k, d] of Object.entries(this.comparison.scores)) {
        if (d.diff !== null && d.diff < -g.maxRegression) {
          fails.push(`${this.name}: ${k} regressed by ${(-d.diff).toFixed(4)} (> ${g.maxRegression}) vs ${this.comparison.baseline}`);
        }
      }
    }
    if (g.maxErrorRate !== undefined && this.summary.error_rate > g.maxErrorRate) {
      fails.push(`${this.name}: error rate ${(this.summary.error_rate * 100).toFixed(2)}% > ${(g.maxErrorRate * 100).toFixed(2)}%`);
    }
    this.gateFailures = fails;
    return fails;
  }

  toMarkdown(): string {
    const s = this.summary;
    const comp = this.comparison?.scores ?? {};
    const f = (x: number | null) => (x === null ? "-" : x.toFixed(3));
    const lines = [
      `### ${this.name} (${this.experimentId ?? "not stored"})`, "",
      `${s.n_cases} cases x ${s.trials} trials, ${s.errors} errors, latency p50 ${f(s.latency_ms.p50)} ms / p95 ${f(s.latency_ms.p95)} ms`, "",
      "| score | mean | 95% CI | n | vs baseline |", "|---|---|---|---|---|",
    ];
    for (const [k, v] of Object.entries(s.scores)) {
      const c = comp[k];
      const vs = c ? `${c.diff! >= 0 ? "+" : ""}${f(c.diff)} (${c.improved} up / ${c.regressed} down)` : "-";
      lines.push(`| ${k} | ${f(v.mean)} | ${v.ci95 ? `[${f(v.ci95[0])}, ${f(v.ci95[1])}]` : "-"} | ${v.n} | ${vs} |`);
    }
    return lines.join("\n");
  }

  toString(): string {
    return this.toMarkdown();
  }
}

// ---- runner -----------------------------------------------------------------------------------
const isCase = (x: unknown): x is EvalCase =>
  typeof x === "object" && x !== null && !Array.isArray(x) && "input" in x;

function scorerName(s: Scorer<never, never, never>): string {
  return s.scorerName ?? (s.name || "scorer");
}

function normalize(name: string, r: ScorerResult): Score[] {
  if (r === null || r === undefined) return [{ name, score: null }];
  if (typeof r === "boolean") return [{ name, score: r ? 1 : 0 }];
  if (typeof r === "number") return [checked({ name, score: r })];
  if (Array.isArray(r)) return r.map(checked);
  if ("score" in r && "name" in r && typeof (r as Score).name === "string") return [checked(r as Score)];
  return Object.entries(r as Record<string, number | null>).map(([k, v]) => checked({ name: k, score: v }));
}

function checked(s: Score): Score {
  if (s.score !== null && !(s.score >= 0 && s.score <= 1)) throw new RangeError(`score ${s.name}=${s.score} outside [0, 1]`);
  return s;
}

class Semaphore {
  #n: number;
  #waiters: (() => void)[] = [];
  constructor(n: number) { this.#n = n; }
  async run<T>(fn: () => Promise<T>): Promise<T> {
    if (this.#n > 0) this.#n -= 1;
    else await new Promise<void>((r) => this.#waiters.push(r));
    try {
      return await fn();
    } finally {
      const w = this.#waiters.shift();
      if (w) w();
      else this.#n += 1;
    }
  }
}

class TimeoutError extends Error {
  override name = "Timeout";
}

function withTimeout<T>(p: Promise<T>, ms: number | undefined): Promise<T> {
  if (!ms) return p;
  let t: ReturnType<typeof setTimeout>;
  return Promise.race([
    p,
    new Promise<never>((_, rej) => { t = setTimeout(() => rej(new TimeoutError(`task exceeded ${ms / 1000}s`)), ms); }),
  ]).finally(() => clearTimeout(t));
}

/** Registry of results created in this process (the test helpers and CLIs read it). */
export const registry = globalState().registry as EvalResult[];

/**
 * Run an evaluation. Cases run concurrently (bounded); each case is an `eval` span with child
 * `task` and `score` spans. Task errors, timeouts and scorer errors are recorded per case and
 * never abort the run. Results are stored and compared with the baseline experiment.
 */
export async function Eval<I, O, E = unknown>(name: string, o: EvalOptions<I, O, E>): Promise<EvalResult> {
  const tracer = o.tracer ?? getTracer();
  const rows = typeof o.data === "function" ? await o.data() : o.data;
  const cases: EvalCase<I, E>[] = [];
  for (const x of rows) cases.push(isCase(x) ? (x as EvalCase<I, E>) : { input: x as I });
  if (!cases.length) throw new Error(`eval ${JSON.stringify(name)} has no cases`);
  const trials = o.trials ?? 1;
  const sem = new Semaphore(o.maxConcurrency ?? 8);
  const scorers = o.scores ?? [];
  const t0 = performance.now();

  const runOne = (c: EvalCase<I, E>, trial: number): Promise<CaseRecord> => sem.run(async () => {
    const id = c.id ?? caseId(toJsonable(c.input), toJsonable(c.expected ?? null));
    const md = c.metadata ?? {};
    const rec: CaseRecord = {
      case_id: id, trial, input: toJsonable(c.input), expected: toJsonable(c.expected ?? null), metadata: toJsonable(md),
      output: null, scores: {}, score_metadata: {}, error: null, duration_ms: 0, trace_id: "",
    };
    const start = performance.now();
    await tracer.span(name, { type: "eval", input: c.input, expected: c.expected, metadata: { ...md, trial, case_id: id }, parent: null }, async (root) => {
      rec.trace_id = root.traceId;
      let output: O | undefined;
      try {
        output = await tracer.span("task", { type: "task" }, async (ts) => {
          const out = await withTimeout(Promise.resolve().then(() => o.task(c.input, { metadata: md, expected: c.expected, span: root })), o.timeoutMs);
          ts.log({ output: out });
          return out;
        });
        rec.output = toJsonable(output);
      } catch (e) {
        const err = e as Error;
        rec.error = { type: err?.name ?? "Error", message: String(err?.message ?? e).slice(0, 1000) };
        root.status = "error";
      }
      rec.duration_ms = performance.now() - start;
      if (rec.error === null) {
        const args: ScorerArgs<I, O, E> = { input: c.input, output: output as O, expected: c.expected, metadata: md };
        for (const sc of scorers) {
          const sname = scorerName(sc as Scorer<never, never, never>);
          try {
            const res = await tracer.span(sname, { type: "score" }, async (ss) => {
              const r = normalize(sname, await sc(args));
              ss.log({ output: r.map((x) => ({ name: x.name, score: x.score })) });
              return r;
            });
            for (const x of res) {
              rec.scores[x.name] = x.score;
              if (x.metadata && Object.keys(x.metadata).length) rec.score_metadata[x.name] = toJsonable(x.metadata);
            }
          } catch (e) {
            rec.scores[sname] = null;
            (rec.scorer_errors ??= {})[sname] = `${(e as Error)?.name}: ${String((e as Error)?.message ?? e).slice(0, 300)}`;
          }
        }
      }
      root.log({ output: output ?? null, scores: rec.scores });
    });
    return rec;
  });

  const results = await Promise.all(cases.flatMap((c) => Array.from({ length: trials }, (_, t) => runOne(c, t))));
  results.sort((a, b) => (a.case_id < b.case_id ? -1 : a.case_id > b.case_id ? 1 : a.trial - b.trial));
  const summary = summarize(results, trials, (performance.now() - t0) / 1000, o.seed ?? 0);
  summary.metadata = toJsonable(o.metadata ?? {});

  let expId: string | null = null;
  let comparison: Comparison | null = null;
  let path: string | null = null;
  const store = o.storeDir === undefined ? ".evalwright" : o.storeDir;
  if (store !== null) {
    const baseline = o.baseline === undefined ? "latest" : o.baseline;
    const base = baseline === null ? null : await loadExperiment(store, name, baseline);
    expId = newExperimentId();
    path = await saveExperiment(store, name, expId, results, summary);
    if (base) comparison = compareExperiments(base, { id: expId, cases: results });
  }
  await tracer.flush(2000);
  const res = new EvalResult(name, expId, results, summary, comparison, path);
  registry.push(res);
  return res;
}

export function summarize(results: CaseRecord[], trials: number, wallSeconds: number, seed = 0): EvalSummary {
  const names = [...new Set(results.flatMap((r) => Object.keys(r.scores)))].sort();
  const scores: Record<string, ScoreSummary> = {};
  for (const k of names) {
    const vals = results.map((r) => r.scores[k]).filter((v): v is number => v !== null && v !== undefined);
    scores[k] = {
      mean: mean(vals), stderr: stderr(vals), ci95: bootstrapCi(vals, 2000, 0.05, seed), n: vals.length,
      min: vals.length ? Math.min(...vals) : null, max: vals.length ? Math.max(...vals) : null,
    };
  }
  const lat = results.map((r) => r.duration_ms);
  const errors = results.filter((r) => r.error !== null).length;
  return {
    n_cases: new Set(results.map((r) => r.case_id)).size, trials, n_runs: results.length, errors,
    error_rate: errors / results.length,
    scorer_errors: results.reduce((a, r) => a + Object.keys(r.scorer_errors ?? {}).length, 0),
    scores, latency_ms: { p50: percentile(lat, 0.5), p95: percentile(lat, 0.95) }, wall_seconds: wallSeconds,
  };
}

// ---- experiments (on-disk layout shared with the Python SDK: `evalwright compare` reads both) --
const safeName = (name: string) => name.replace(/[^A-Za-z0-9\-_.]/g, "_");

export function newExperimentId(): string {
  const d = new Date();
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
    + `${p(d.getMilliseconds(), 3)}-${Math.floor(Math.random() * 65536).toString(16).padStart(4, "0")}`;
}

export async function saveExperiment(store: string, name: string, id: string, cases: CaseRecord[], summary: EvalSummary): Promise<string> {
  const dir = join(store, safeName(name), id);
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "cases.jsonl"), cases.map((c) => JSON.stringify(c)).join("\n") + "\n", "utf8");
  const createdNs = String(BigInt(Date.now()) * 1_000_000n);
  await writeFile(join(dir, "summary.json"), JSON.stringify({ ...summary, created_ns: Number(createdNs) }, null, 2));
  const tmp = join(store, safeName(name), `.latest.${process.pid}`);
  await writeFile(tmp, id);
  await rename(tmp, join(store, safeName(name), "LATEST"));     // atomic pointer update
  return dir;
}

async function listExperiments(store: string, name: string): Promise<string[]> {
  const d = join(store, safeName(name));
  let entries: string[];
  try {
    entries = await readdir(d);
  } catch {
    return [];
  }
  const found: [number, string][] = [];
  for (const e of entries) {
    try {
      const s = JSON.parse(await readFile(join(d, e, "summary.json"), "utf8")) as { created_ns?: number };
      found.push([Number(s.created_ns ?? 0), e]);
    } catch { /* not an experiment dir */ }
  }
  return found.sort((a, b) => a[0] - b[0]).map(([, e]) => e);
}

export async function loadExperiment(store: string, name: string, ref = "latest"): Promise<{ id: string; cases: CaseRecord[]; summary: EvalSummary } | null> {
  let id: string | null = null;
  if (ref === "latest") {
    try { id = (await readFile(join(store, safeName(name), "LATEST"), "utf8")).trim(); } catch { id = null; }
  } else if (ref === "previous") {
    const ids = await listExperiments(store, name);
    id = ids.length >= 2 ? ids[ids.length - 2]! : null;
  } else {
    id = ref;
  }
  if (!id) return null;
  try {
    const dir = join(store, safeName(name), id);
    const cases = (await readFile(join(dir, "cases.jsonl"), "utf8")).split("\n").filter(Boolean).map((l) => JSON.parse(l) as CaseRecord);
    const summary = JSON.parse(await readFile(join(dir, "summary.json"), "utf8")) as EvalSummary;
    return { id, cases, summary };
  } catch {
    return null;
  }
}

function perCase(cases: CaseRecord[]): Map<string, Map<string, number>> {
  const acc = new Map<string, Map<string, number[]>>();
  for (const c of cases) {
    for (const [k, v] of Object.entries(c.scores ?? {})) {
      if (v === null) continue;
      const m = acc.get(c.case_id) ?? new Map<string, number[]>();
      m.set(k, [...(m.get(k) ?? []), v]);
      acc.set(c.case_id, m);
    }
  }
  const out = new Map<string, Map<string, number>>();
  for (const [cid, m] of acc) out.set(cid, new Map([...m].map(([k, vs]) => [k, mean(vs)!])));
  return out;
}

/** Paired, case-by-case comparison on cases present in both runs. */
export function compareExperiments(baseline: { id: string; cases: CaseRecord[] }, current: { id: string; cases: CaseRecord[] }, eps = 1e-9): Comparison {
  const b = perCase(baseline.cases);
  const c = perCase(current.cases);
  const common = [...c.keys()].filter((k) => b.has(k)).sort();
  const scorers = [...new Set([...[...c.values()].flatMap((m) => [...m.keys()]), ...[...b.values()].flatMap((m) => [...m.keys()])])].sort();
  const out: Comparison = {
    baseline: baseline.id, current: current.id, common_cases: common.length,
    new_cases: [...c.keys()].filter((k) => !b.has(k)).length, missing_cases: [...b.keys()].filter((k) => !c.has(k)).length, scores: {},
  };
  for (const s of scorers) {
    const pairs = common.filter((i) => b.get(i)!.has(s) && c.get(i)!.has(s)).map((i) => [b.get(i)!.get(s)!, c.get(i)!.get(s)!] as const);
    if (!pairs.length) continue;
    const diffs = pairs.map(([x, y]) => y - x);
    out.scores[s] = {
      baseline: mean(pairs.map(([x]) => x)), current: mean(pairs.map(([, y]) => y)), diff: mean(diffs), ci95: bootstrapCi(diffs),
      improved: diffs.filter((d) => d > eps).length, regressed: diffs.filter((d) => d < -eps).length,
      unchanged: diffs.filter((d) => Math.abs(d) <= eps).length, n: diffs.length,
    };
  }
  return out;
}
