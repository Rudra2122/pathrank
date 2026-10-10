# Changelog

All notable changes to `evalwright`. Follows [Semantic Versioning](https://semver.org): until 1.0,
minor versions may change public APIs, always with a deprecation warning one minor release ahead.

## [0.1.0] - 2026-10-09
### Added
- Wire protocol v1 (`sdk/spec`): JSON Schema for span records, content-derived `Idempotency-Key`
  on every HTTP export (identical across retries), shared with the TypeScript SDK.
- `evalwright serve`: local collector with schema validation, idempotency dedupe and fault
  injection (`--fail-rate`, `--fail-status`, `--retry-after`) for testing any SDK.
- Testing framework: `evalwright.testing` (`capture_spans`, `SpanCapture.tree()`, `assert_eval`) and a
  pytest plugin (`pytest_evalwright`, auto-registered): `ew_spans` fixture, `--ew-store-dir`,
  `--ew-no-baseline`, `--ew-report`, end-of-session eval summary. The plugin imports the SDK lazily.
- Profiling: `@traced(profile=True|N)` and `with evalwright.profiled():` attach a cProfile top-N
  report to the span; sampled spans only, nested-safe, yields to other active profilers.
- Tracing: `init`, `span` (sync + async context manager), `@traced` for sync/async/generator/async-generator
  functions, `current_span`, `log`, `run_in_thread_context`; ContextVar propagation across asyncio tasks.
- Safe serialization (`to_jsonable`): total (never raises), bounded (depth/string/items), cycle-safe,
  numpy/pydantic/dataclass aware; default secret redaction, optional email redaction.
- Deterministic per-trace sampling; no-op fast path for disabled or sampled-out traces.
- Background `BatchProcessor`: bounded drop-oldest queue, batching, exponential backoff with full jitter,
  `Retry-After`, circuit breaker, fork safety, `flush`/`shutdown`, stats.
- Exporters: JSONL, HTTP (gzip, auth), OTLP/HTTP JSON (OpenTelemetry), console, in-memory.
- Evals: `Eval` / `eval_async` with concurrency limit, trials, timeouts, per-case error capture,
  scorer kwargs introspection, bootstrap CIs, local experiments, paired comparison vs baseline, gates.
- Scorers: ExactMatch, Contains, JSONValid, Levenshtein, NumericCloseness, RegexMatch, LLMJudge (model-agnostic).
- CLI: `evalwright run | compare | ls` with `--min`, `--max-regression`, `--max-error-rate` CI gates.
