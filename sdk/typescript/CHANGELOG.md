# Changelog

All notable changes to the `evalwright` TypeScript SDK. Follows [Semantic Versioning](https://semver.org).

## [0.1.0] - 2026-10-09
### Added
- Tracing: `init`, `span` (sync/async callbacks), `traced` (sync, async, generator, async-generator),
  `currentSpan`, `log`, `flush`; `AsyncLocalStorage` context; deterministic per-trace sampling
  (CRC-32, identical to the Python SDK).
- `toJsonable`: total, bounded, cycle-safe serialization with default secret redaction.
- `BatchProcessor`: bounded drop-oldest queue, batching, retries with full jitter and `Retry-After`,
  circuit breaker, flush with timeout, `beforeExit` flush, unref'd timers.
- Exporters: HTTP (gzip, auth, content-derived `Idempotency-Key`), OTLP/HTTP JSON, JSONL, console, in-memory.
- Evals: `Eval` with concurrency limit, trials, timeouts, per-case error capture, bootstrap CIs, experiment
  storage and comparison shared with the Python SDK, gates.
- Scorers: exactMatch, contains, jsonValid, levenshtein, numericCloseness, regexMatch, llmJudge.
- `evalwright/testing`: `captureSpans`, `assertEval`.
- Global state shared across package copies (ESM + CJS) via `globalThis`.
