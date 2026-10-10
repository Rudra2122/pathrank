# evalwright SDKs

Tracing and evaluation for LLM applications, in two languages that speak one protocol.

| | Python | TypeScript |
|---|---|---|
| package | [`sdk/python`](python) (`pip install evalwright`) | [`sdk/typescript`](typescript) (`npm install evalwright`) |
| runtimes | CPython 3.10–3.13 | Node 18.18+ (ESM + CJS) |
| runtime dependencies | none | none |
| typing | mypy `--strict` | `strict`, `exactOptionalPropertyTypes` |
| context | `ContextVar` | `AsyncLocalStorage` |
| testing helpers | `evalwright.testing` + pytest plugin | `evalwright/testing` |
| CLI | `evalwright run / compare / ls / serve` | uses the Python CLI (shared experiment files) |

[`spec/`](spec) defines the contract both implement: the span record schema and the HTTP ingestion
protocol (batching, gzip, idempotency keys, retry semantics). A cross-language contract test runs
the built TypeScript package and the Python SDK against the same collector, through injected
failures, and validates every record against the schema.

## Layout

```
spec/        span.schema.json, PROTOCOL.md
python/      src/evalwright (SDK), src/pytest_evalwright (plugin), tests, benchmarks
typescript/  src (SDK), tests (vitest), scripts/smoke-pack.sh, scripts/contract.mjs
```

## CI/CD (.github/workflows)

| workflow | what it gates |
|---|---|
| `sdk-python-ci` | ruff, mypy --strict, tests on 3 OSes x Python 3.10–3.13, coverage >= 90%, wheel built and tested **as installed** (zero deps), overhead budget |
| `sdk-typescript-ci` | typecheck, vitest + coverage thresholds on 3 OSes x Node 22/24, build, publint + are-the-types-wrong, packed tarball exercised on Node 18/20/22/24 |
| `sdk-contract` | both SDKs against one collector with injected failures; schema validation; cross-language experiment comparison |
| `sdk-python-release` | tag `sdk-py-vX.Y.Z`: version/CHANGELOG check, TestPyPI, install smoke test, PyPI via trusted publishing |
| `sdk-typescript-release` | tag `sdk-ts-vX.Y.Z`: version/CHANGELOG check, full gate, npm publish with provenance |
