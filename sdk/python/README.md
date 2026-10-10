# evalwright

Tracing and evaluation for LLM applications. Zero runtime dependencies, fully typed (mypy
`--strict`), Python 3.10–3.13. PathRank's notification agent and API use it (see the repo root).

```bash
pip install evalwright            # or, from this repo: pip install ./sdk/python
```

## Tracing

```python
import evalwright as ew

ew.init("my-app")        # JSONL by default; EVALWRIGHT_EXPORT=otlp:http://localhost:4318/v1/traces for OpenTelemetry

@ew.traced(type="llm")
async def generate(prompt: str) -> str:
    ew.log(metrics={"prompt_tokens": 812}, metadata={"model": "..."})
    return await client.complete(prompt)

async with ew.span("answer", type="task", input=question) as s:
    docs = retrieve(question)                 # @traced spans nest automatically
    s.log(output=await generate(build_prompt(question, docs)))
```

- `span()` works as `with` and `async with`; `@traced` handles sync, async, generator and async-generator
  functions. The current span lives in a `ContextVar`, so concurrent asyncio tasks never mix spans;
  for thread pools wrap work in `ew.run_in_thread_context(fn)`.
- Generators only hold their span while their body runs, so a suspended generator never leaks its
  span into the consumer's context (a common tracing bug, covered by a test).
- Every span records wall time, CPU time, status, and the exception type, message and stack on error.
  Exceptions are recorded and re-raised, never swallowed.
- Sampling is per trace and deterministic (hash of the trace id): whole traces are kept or dropped.

## Evals

```python
from evalwright import Eval
from evalwright.scorers import ExactMatch, Levenshtein, LLMJudge

result = Eval("support-bot", data=cases, task=answer,
              scores=[ExactMatch(), Levenshtein(), LLMJudge(my_llm_fn, rubric="Is it correct and polite?")],
              max_concurrency=16, trials=3, timeout=30)
print(result)                         # per-score mean, bootstrap 95% CI, comparison vs last run
```

- Cases run concurrently (async tasks directly, sync tasks in worker threads) under a semaphore; each is
  an `eval` span with child `task` and `score` spans, so every score links to its trace.
- Task errors, timeouts and scorer errors are recorded per case. They never abort the run.
- Scorers are plain callables taking any subset of `input, output, expected, metadata` and returning a float
  in [0, 1], a bool, a `Score`, a list of Scores, a dict, or `None` (not applicable).
- Each run is stored as an experiment (`.evalwright/<name>/<id>/`) and compared case-by-case with the
  previous one: mean diff, bootstrap CI, improved/regressed counts.

### In CI

```bash
evalwright run evals/ --max-regression 0.02 --min exact_match=0.8 --max-error-rate 0.05 --json report.json
```

Exit code 0 = pass, 1 = a gate failed, 2 = an eval file failed to import. `evalwright compare <name>`
and `evalwright ls` inspect stored experiments.

## Testing your app

```python
from evalwright.testing import capture_spans, assert_eval

def test_agent_traces():
    with capture_spans() as cap:
        run_agent("hi")
    assert cap.tree() == "agent [task]\n  draft [llm]\n  guardrails [function]"
    assert cap.find("draft").metrics["tokens"] > 0

def test_agent_quality():
    assert_eval(Eval("agent", data, task, scores), min={"judge": 0.8}, max_regression=0.02)
```

The pytest plugin is registered automatically: the `ew_spans` fixture is a span capture,
`--ew-store-dir` / `--ew-no-baseline` control eval storage in tests, `--ew-report out.json` writes
every eval run during the session, and the terminal report ends with an eval summary.

## Profiling

```python
@ew.traced(profile=True)          # or profile=10 for the top 10 functions
def rank(user): ...

with ew.span("batch"), ew.profiled():
    run_batch()
```

Sampled spans get `metadata["profile"]`: the top functions by cumulative time. Nested profiled spans
are covered by the outermost one; if another profiler already owns the thread, profiling is skipped
rather than raising. `benchmarks/overhead.py` measures the SDK's own per-call overhead, and CI
fails if it exceeds `benchmarks/budget.json`.

## Local collector

`evalwright serve --port 8787 --fail-rate 0.3` runs a collector for wire protocol v1
([`sdk/spec/PROTOCOL.md`](../spec/PROTOCOL.md)): validates records, dedupes retried batches by
`Idempotency-Key`, and injects failures so you can watch retries and the circuit breaker work.

## Resilience guarantees

Tracing must never take down or slow down the application it observes:

| failure | behavior |
|---|---|
| exporter endpoint down / 5xx / 429 / timeout | retried with exponential backoff + full jitter, `Retry-After` honored |
| 4xx (bad payload, auth) | batch dropped and counted (retrying can't help) |
| endpoint failing repeatedly | circuit breaker opens, exports pause for a cooldown, one probe batch, then recover |
| producer faster than the network | bounded queue, oldest spans dropped and counted; the caller never blocks |
| unserializable / huge / cyclic objects logged | `to_jsonable` is total and bounded: depth, string length and item caps |
| exporter raises | treated as retryable; the worker thread survives |
| retry after a request that actually landed | content-derived `Idempotency-Key`, so the collector can dedupe |
| secrets in inputs | values under secret-looking keys and `sk-…`/`Bearer …`/AWS keys masked by default |
| `os.fork()` | child gets a fresh queue, lock and worker; parent's pending spans are not duplicated |
| process exit | `atexit` flush with a timeout; `flush(timeout)` returns whether everything was delivered |

`tracer.stats()` exposes enqueued, exported, dropped (queue full / failed), retries, circuit state, and last error.

## Performance

Measured with `benchmarks/overhead.py` (CPython 3.13, 2-vCPU cloud VM). CI fails if overhead exceeds
`benchmarks/budget.json`.

| case | overhead per call |
|---|---|
| `@traced`, tracing disabled | ~1.3 µs |
| `@traced`, trace sampled out | ~2.2 µs |
| `@traced`, exported (small args/result) | ~19 µs |
| three nested spans | ~26 µs |

The profiler drove three fixes in 0.1.0: a shared no-op span for disabled/sampled-out traces (19 → 1.3 µs),
resolving `inspect.signature` once per decorated function instead of per call, and a scalar fast path plus
cached key-redaction in the serializer (50 → 19 µs exported). Serialization deliberately stays on the caller
thread: deferring it to the background thread would be faster but would record mutated data if user code
changes a logged object after the call returns.

## Exporters

`JsonlExporter(path)`, `HttpExporter(url, api_key)` (gzip JSON), `OtlpHttpExporter(endpoint)` (OpenTelemetry
OTLP/HTTP JSON: Jaeger, Tempo, any OTel collector), `ConsoleExporter()`, `InMemoryExporter()` (tests).
Custom: any object with `export(batch) -> ExportResult` and `close()`.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && mypy && pytest --cov            # coverage gate: 90%
python benchmarks/overhead.py --profile --check benchmarks/budget.json
python -m build && twine check dist/*
```

CI (`.github/workflows/sdk-ci.yml`): lint, strict typing, tests on Linux/macOS/Windows × Python 3.10–3.13,
coverage gate, builds the wheel and runs the test suite against the *installed* wheel (catches files missing
from the package), verifies zero runtime dependencies, runs the overhead budget.
Release (`sdk-release.yml`): push tag `sdk-vX.Y.Z` → checks tag = version and a CHANGELOG entry exists → build →
TestPyPI → install from TestPyPI → PyPI (trusted publishing, no stored tokens) → GitHub release.
