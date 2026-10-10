# evalwright (TypeScript)

Tracing and evaluation for LLM applications. Zero runtime dependencies, strict TypeScript,
ESM + CommonJS, Node 18.18+. Wire-compatible with the Python SDK: same span records, same
sampling decisions, same experiment files (see [`sdk/spec`](../spec)).

```bash
npm install evalwright
```

## Tracing

```ts
import * as ew from "evalwright";

ew.init("my-app");   // JSONL by default; EVALWRIGHT_EXPORT=otlp:http://localhost:4318/v1/traces for OpenTelemetry

const retrieve = ew.traced(async (q: string) => search(q), { name: "retrieve", type: "retrieval" });
const generate = ew.traced(async (prompt: string) => {
  ew.log({ metrics: { prompt_tokens: 812 }, metadata: { model: "..." } });
  return client.complete(prompt);
}, { name: "generate", type: "llm" });

const answer = await ew.span("answer", { type: "task", input: question }, async (span) => {
  const out = await generate(build(question, await retrieve(question)));   // nests automatically
  span.log({ output: out });
  return out;
});
```

- Context lives in `AsyncLocalStorage`, so concurrent requests never mix spans, across `await`,
  timers, and `Promise.all`.
- `span()` takes sync or async callbacks and ends the span when the promise settles. `traced()`
  wraps sync, async, generator and async-generator functions, keeps `this`, `name` and the type
  signature. Generators hold their span only while running, never while suspended.
- Errors are recorded (type, message, stack) and rethrown, never swallowed.
- Without `init()`, every call works and does nothing, so libraries can instrument unconditionally.
- Sampling is per trace and deterministic (CRC-32 of the trace id), identical to the Python SDK: a
  trace crossing a Python service and a Node service is kept or dropped as a whole.

## Evals

```ts
import { Eval, scorers } from "evalwright";

const result = await Eval("support-bot", {
  data: cases,                                   // [{ input, expected?, metadata?, id? }] or plain inputs
  task: (input) => answer(input),
  scores: [scorers.exactMatch(), scorers.levenshtein(), scorers.llmJudge(myLlm, { rubric: "Correct and polite?" })],
  maxConcurrency: 16, trials: 3, timeoutMs: 30_000,
});
console.log(result.toMarkdown());               // means, bootstrap 95% CIs, comparison vs the last run
```

Task errors, timeouts and scorer errors are recorded per case and never abort a run. Each run is
stored in `.evalwright/<name>/<id>/` using the same layout as the Python SDK, so
`evalwright compare <name>` (Python CLI) compares TypeScript runs too, and case ids match across
languages for the same inputs.

## Testing

```ts
import { captureSpans, assertEval } from "evalwright/testing";

const cap = captureSpans();
await runAgent("hi");
expect(await cap.tree()).toBe("agent [task]\n  draft [llm]\n  guardrails [function]");
cap.restore();

assertEval(await Eval("agent", { data, task, scores }), { min: { judge: 0.8 }, maxRegression: 0.02 });
```

Works with vitest, jest, `node:test`, or anything that treats a thrown error as a failure.

## Resilience

The same guarantees as the Python SDK, from the shared [protocol](../spec/PROTOCOL.md):

| failure | behavior |
|---|---|
| endpoint down / 5xx / 429 / timeout | retried with exponential backoff + full jitter; `Retry-After` honored |
| other 4xx | batch dropped and counted |
| endpoint failing repeatedly | circuit breaker opens, pauses, then probes |
| retry after a request that actually landed | content-derived `Idempotency-Key` lets the collector dedupe |
| producer faster than the network | bounded queue, oldest dropped and counted; callers never block |
| unserializable / cyclic / huge values | `toJsonable` is total and bounded |
| secrets in inputs | masked by default (`sk-…`, `Bearer …`, AWS keys, secret-looking keys) |
| process exit | `beforeExit` flush with a timeout; background timers are `unref`'d and never keep a process alive |
| ESM and CJS copies loaded together | process state lives on `globalThis` under a versioned `Symbol.for` key and is shared |

## Development

```bash
npm ci
npm run typecheck && npm test && npm run build && npm run check:package
bash scripts/smoke-pack.sh      # install the packed tarball into an empty project, import ESM + CJS
```
