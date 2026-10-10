// Cross-language contract driver: emits spans and evals with the BUILT TypeScript package so the
// Python test suite (sdk/python/tests/test_contract.py) can validate them against sdk/spec.
//   EVALWRIGHT_URL=http://127.0.0.1:8787 EVALWRIGHT_STORE=/tmp/store node scripts/contract.mjs
import * as ew from "../dist/index.js";

const url = process.env.EVALWRIGHT_URL;
if (!url) throw new Error("EVALWRIGHT_URL is required");
const tracer = ew.init("contract-ts", {
  exporter: new ew.HttpExporter(`${url}/v1/spans`),
  retry: new ew.RetryPolicy({ maxRetries: 25, baseDelayMs: 1, maxDelayMs: 20 }),
  maxBatch: 5,
  flushIntervalMs: 10,
});

const tool = ew.traced(async (q) => ({ docs: [q], api_key: "sk-ABCDEFGHIJKLMNOPQRSTUVWX" }), { name: "retrieve", type: "retrieval" });
const llm = ew.traced(async (prompt) => {
  ew.log({ metrics: { prompt_tokens: 12, completion_tokens: 3 }, metadata: { model: "test" }, tags: ["llm", "test"] });
  return `answer to ${prompt}`;
}, { name: "generate", type: "llm" });

for (let i = 0; i < 10; i++) {
  await ew.span("answer", { type: "task", input: { q: i } }, async (s) => {
    const ctx = await tool(`q${i}`);
    s.log({ output: await llm(JSON.stringify(ctx)), scores: { relevance: i / 10 } });
  });
}
try {
  await ew.span("failing", async () => { throw new TypeError("expected failure"); });
} catch { /* recorded on the span */ }

const store = process.env.EVALWRIGHT_STORE;
if (store) {
  const data = Array.from({ length: 8 }, (_, i) => ({ input: i, expected: i * 2 }));
  await ew.Eval("ts-contract-eval", { data, task: (x) => x * 2, scores: [ew.scorers.exactMatch()], storeDir: store });
  await ew.Eval("ts-contract-eval", { data, task: (x) => (x < 4 ? x * 2 : -1), scores: [ew.scorers.exactMatch()], storeDir: store });
}
const ok = await tracer.flush(20000);
console.log(JSON.stringify({ flushed: ok, stats: tracer.stats() }));
process.exit(ok ? 0 : 1);
