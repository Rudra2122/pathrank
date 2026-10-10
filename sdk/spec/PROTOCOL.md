# evalwright wire protocol v1

Both SDKs (Python, TypeScript) speak this protocol. A cross-language contract test
(`sdk/python/tests/test_contract.py`) runs both against the same collector and validates every
record against `span.schema.json`.

## Records

One finished span = one JSON object matching [`span.schema.json`](span.schema.json). The SDK
makes values JSON-safe before sending: bounded depth, string length and item counts; cycles cut;
secret-looking values masked. A collector never has to defend against unbounded payloads from a
well-behaved SDK.

## HTTP ingestion

```
POST {base_url}/v1/spans
Content-Type: application/json
Content-Encoding: gzip                (optional; SDKs gzip by default)
Authorization: Bearer <api key>       (optional)
Idempotency-Key: <32 hex chars>       (same value on every retry of the same batch)
User-Agent: evalwright-<language>/<sdk version> <runtime>/<version>

{"spans": [<record>, ...]}
```

`Idempotency-Key` is the first 32 hex chars of SHA-256 over the batch's span ids joined by `\n`.
It's derived from content, not random, so a retry after a timeout that actually reached the
server is recognizable as a duplicate. Collectors should dedupe on it (the dev collector does).

## Responses and client behavior

| response | meaning | SDK behavior |
|---|---|---|
| 2xx | accepted | done |
| 408, 425, 429, 5xx, network error, timeout | transient | retry with exponential backoff and full jitter; honor `Retry-After` (seconds); up to `max_retries` |
| other 4xx (400, 401, 403, 413, ...) | the request itself is wrong | drop the batch and count it; retrying can't help |
| repeated failures | endpoint unhealthy | circuit breaker opens: exports pause for a cooldown, then a single probe batch |

## Guarantees every SDK must keep

1. Never raise into, block, or crash the host application because of tracing.
2. Bounded memory: a fixed-size queue; when full, drop the oldest spans and count the drops.
3. `flush(timeout)` returns whether everything queued was delivered; exit hooks flush with a timeout.
4. Without `init()`, every API works and does nothing (libraries can instrument unconditionally).
5. Sampling is decided once per trace, deterministically from the trace id, so traces are never partial.

## Dev collector

`evalwright serve --port 8787 [--fail-rate 0.3] [--fail-status 503] [--out spans.jsonl]` runs a local
collector implementing this protocol: schema checks, idempotency dedupe, and fault injection for
testing retry behavior. `GET /v1/spans` returns everything received; `GET /v1/stats` returns counters.
