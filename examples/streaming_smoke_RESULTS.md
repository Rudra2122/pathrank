# Streaming smoke test (synthetic, no Kafka)

`make stream-smoke` on a 2-core Linux VM: 300K synthetic OLX-shaped events, Kafka-shaped Parquet files instead of a broker, local lake instead of MinIO. This checks correctness (both parity checks), not performance; the Spark timing is mostly JVM/job startup on tiny data. Real numbers: README streaming table, from `make stream-bench` on the 65.5M OLX events.

| Metric | Result |
|---|---|
| Producer throughput (Java, max rate) | not run |
| End-to-end lag, Kafka -> S3 (p50 / p95) | n/a: file source, no Kafka (300,000 events in 6 micro-batches) |
| Spark batch vs pandas, same features + data | **Spark 9.1 s vs pandas 0.3 s** (0.03x; 89,182 user-job pairs, Spark local[*] on 2 cores) |
| Feature parity, Spark vs pandas `AsOf` | **max diff 0** over job_pop_all, job_pop_7d, user_n_apps, user_last (26,448 users, 6,121 jobs, cutoff 2020-02-24 00:00:00); PASS |
| Stream vs batch windows (1 h) | max diff 0; 40,331 job windows + 240,213 user windows matched; 0 closed windows missing; PASS |
| Kafka -> S3 completeness | 300,000 rows landed / 300,000 produced, 0 duplicates |

## stream.json

```json
{
 "source": "files",
 "topic": "job-events",
 "trigger": "2 seconds",
 "window": "1 hour",
 "watermark": "1 hour",
 "max_offsets_per_trigger": 2000000,
 "sink": "local",
 "producer_rate": null,
 "spark": "4.0.1",
 "cores": 2,
 "events": 300000,
 "batches": 6,
 "lag_ms": {
  "p50": 25867,
  "p95": 31354,
  "p99": 31354,
  "max": 31354,
  "mean": 26200.7
 },
 "wall_s": 20.7,
 "rows_per_s": 14495
}
```

## batch_parity.json

```json
{
 "events_parquet": "data/streaming_smoke/events.parquet",
 "reference": "rebuild",
 "cutoff": "2020-02-24 00:00:00",
 "pairs": 89182,
 "pairs_spark": 89182,
 "users": 26448,
 "jobs": 6121,
 "jobs_check": {
  "only_pandas": 0,
  "only_spark": 0,
  "pop_all_nan_mismatch": 0,
  "pop_all_max_abs_diff": 0.0,
  "pop_7d_nan_mismatch": 0,
  "pop_7d_max_abs_diff": 0.0
 },
 "users_check": {
  "only_pandas": 0,
  "only_spark": 0,
  "n_apps_nan_mismatch": 0,
  "n_apps_max_abs_diff": 0.0,
  "last_days_nan_mismatch": 0,
  "last_days_max_abs_diff": 0.0
 },
 "max_abs_diff": 0.0,
 "row_or_nan_mismatches": 0,
 "pass": true,
 "pandas_s": {
  "read_s": 0.11,
  "schema_s": 0.14,
  "asof_s": 0.02,
  "total_s": 0.27
 },
 "spark_s": 9.11,
 "speedup": 0.03,
 "spark": {
  "version": "4.0.1",
  "master": "local[*]",
  "cores": 2
 },
 "machine": {
  "platform": "Linux-6.18.44-fc-v114-x86_64-with-glibc2.39",
  "machine": "x86_64",
  "cpus": 2
 }
}
```

## stream_parity.json

```json
{
 "events_parquet": "data/streaming_smoke/events.parquet",
 "events_in_parquet": 300000,
 "window": "1 hour",
 "watermark": "1 hour",
 "closed_by": "2020-02-25 22:59:57",
 "job_popularity": {
  "matched": 40331,
  "stream_only": 0,
  "open_at_end": 247,
  "missing_closed": 0,
  "events_in_open_windows": 1778,
  "batch_or_stream_windows": 40578,
  "events_max_abs_diff": 0,
  "contacts_max_abs_diff": 0
 },
 "user_activity": {
  "matched": 240213,
  "stream_only": 0,
  "open_at_end": 1444,
  "missing_closed": 0,
  "events_in_open_windows": 1778,
  "batch_or_stream_windows": 241657,
  "events_max_abs_diff": 0,
  "contacts_max_abs_diff": 0,
  "last_ts_max_abs_diff": 0
 },
 "bronze": {
  "rows": 300000,
  "duplicates": 0
 },
 "max_abs_diff": 0,
 "pass": true
}
```
