"""Spark Structured Streaming job: Kafka `job-events` -> S3 (MinIO).

Three queries on the same topic, each with its own checkpoint:
  bronze          raw events -> s3a://pathrank/bronze/job_events/batch_id=N/  (foreachBatch)
                  After each batch's Parquet write returns, lag = commit time - Kafka record timestamp
                  is recorded for every row (exact histogram, 1 ms buckets). Writes overwrite their
                  batch_id folder, so a retried batch replaces itself instead of duplicating rows.
  job_popularity  per job and 1 h event-time window: events, contact events  (append + watermark)
  user_activity   per user and 1 h window: events, contact events, last event ts

job_popularity() / user_activity() are plain DataFrame functions, so batch.py runs the exact same
code over the full Parquet and parity.py compares the two.
"""
from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path

from ..utils import log
from . import CONTACT_EVENTS
from .spark import lake, session

EVENT_DDL = "user LONG, item LONG, event STRING, ts LONG"
# what the Kafka source returns (minus topic/timestampType/headers); the file source in tests uses it too
KAFKA_LIKE_DDL = "key BINARY, value BINARY, timestamp TIMESTAMP, partition INT, offset LONG"


def parse(raw):
    from pyspark.sql import functions as F
    e = F.from_json(F.col("value").cast("string"), EVENT_DDL)
    return (raw.select(e.alias("e"), F.col("timestamp").alias("kafka_ts"), "partition", "offset")
            .select("e.*", "kafka_ts", "partition", "offset")
            .withColumn("event_time", F.timestamp_seconds("ts")))


def _contact():
    from pyspark.sql import functions as F
    return F.col("event").isin(list(CONTACT_EVENTS)).cast("long")


def job_popularity(ev, window: str, watermark: str | None = None):
    from pyspark.sql import functions as F
    if watermark:
        ev = ev.withWatermark("event_time", watermark)
    return (ev.groupBy(F.window("event_time", window).alias("w"), "item")
            .agg(F.count(F.lit(1)).alias("events"), F.sum(_contact()).alias("contacts"))
            .select(F.col("w.start").alias("window_start"), F.col("w.end").alias("window_end"),
                    "item", "events", "contacts"))


def user_activity(ev, window: str, watermark: str | None = None):
    from pyspark.sql import functions as F
    if watermark:
        ev = ev.withWatermark("event_time", watermark)
    return (ev.groupBy(F.window("event_time", window).alias("w"), "user")
            .agg(F.count(F.lit(1)).alias("events"), F.sum(_contact()).alias("contacts"),
                 F.max("ts").alias("last_ts"))
            .select(F.col("w.start").alias("window_start"), F.col("w.end").alias("window_end"),
                    "user", "events", "contacts", "last_ts"))


def percentiles(hist: Counter, qs=(0.5, 0.95, 0.99)) -> dict:
    """Nearest-rank percentiles from an exact value -> count histogram."""
    tot = sum(hist.values())
    if not tot:
        return {}
    out, cum, keys = {}, 0, sorted(hist)
    targets = sorted(qs)
    i = 0
    for k in keys:
        cum += hist[k]
        while i < len(targets) and cum >= targets[i] * tot:
            out[f"p{round(targets[i] * 100):d}"] = k
            i += 1
    out["max"] = keys[-1]
    out["mean"] = round(sum(k * v for k, v in hist.items()) / tot, 1)
    return out


class BronzeSink:
    """foreachBatch: write the batch to S3, then record per-row Kafka -> S3 lag."""

    def __init__(self, path: str, out_json: Path, meta: dict):
        self.path, self.out_json, self.meta = path, out_json, meta
        self.hist: Counter = Counter()
        self.batches: list[dict] = []
        self.rows = 0
        self.t_first: float | None = None

    def __call__(self, df, batch_id: int) -> None:
        from pyspark.sql import functions as F
        t0 = time.time()
        df = df.persist()
        n = df.count()
        if n == 0:
            df.unpersist()
            return
        self.t_first = self.t_first or t0
        df.drop("event_time").write.mode("overwrite").parquet(f"{self.path}/batch_id={batch_id}")
        commit_ms = int(time.time() * 1000)            # the Parquet files are in S3 now
        lag = (df.select((F.lit(commit_ms) - F.unix_millis("kafka_ts")).alias("lag"))
               .groupBy("lag").count().collect())
        df.unpersist()
        for r in lag:
            self.hist[int(r["lag"])] += int(r["count"])
        self.rows += n
        self.batches.append({"batch_id": batch_id, "rows": n, "write_s": round(time.time() - t0, 3),
                             "commit_ms": commit_ms})
        log.info("bronze batch %d: %s rows in %.1fs (total %s)", batch_id, f"{n:,}", time.time() - t0,
                 f"{self.rows:,}")
        self.flush()

    def summary(self) -> dict:
        wall = (self.batches[-1]["commit_ms"] / 1000 - self.t_first) if self.batches else 0.0
        return {**self.meta, "events": self.rows, "batches": len(self.batches),
                "lag_ms": percentiles(self.hist), "wall_s": round(wall, 2),
                "rows_per_s": round(self.rows / wall) if wall > 0 else None,
                "per_batch": self.batches}

    def flush(self) -> None:
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        self.out_json.write_text(json.dumps(self.summary(), indent=1))


def rm(spark, uri: str) -> None:
    jvm = spark._jvm
    p = jvm.org.apache.hadoop.fs.Path(uri)
    p.getFileSystem(spark._jsc.hadoopConfiguration()).delete(p, True)


def run(cfg, source: str = "kafka", files_dir: str | None = None, idle_s: float = 30,
        expect: int | None = None, reset: bool = False, max_wait_s: float = 4 * 3600,
        producer_rate: int | None = None) -> dict:
    s = cfg.streaming
    (Path(s.results_dir) / "stream.ready").unlink(missing_ok=True)
    spark = session(cfg, "pathrank-stream", kafka=source == "kafka")
    ck = Path(s.checkpoint_dir)
    paths = {"bronze": lake(cfg, "bronze", "job_events"),
             "job_popularity": lake(cfg, "features", "job_popularity_1h"),
             "user_activity": lake(cfg, "features", "user_activity_1h")}
    if reset:
        shutil.rmtree(ck, ignore_errors=True)
        for p in paths.values():
            rm(spark, p)
        log.info("reset: removed checkpoints and lake outputs")

    if source == "kafka":
        raw = (spark.readStream.format("kafka")
               .option("kafka.bootstrap.servers", s.bootstrap)
               .option("subscribe", s.topic)
               .option("startingOffsets", "earliest")
               .option("maxOffsetsPerTrigger", s.max_offsets_per_trigger)
               .load())
    else:   # tests / no-Kafka smoke: Parquet files shaped like Kafka rows
        raw = spark.readStream.schema(KAFKA_LIKE_DDL).option("maxFilesPerTrigger", 1).parquet(files_dir)
    ev = parse(raw)

    meta = {"source": source, "topic": s.topic, "trigger": s.trigger, "window": s.window,
            "watermark": s.watermark, "max_offsets_per_trigger": s.max_offsets_per_trigger,
            "sink": s.sink, "producer_rate": producer_rate, "spark": spark.version,
            "cores": spark.sparkContext.defaultParallelism}
    sink = BronzeSink(paths["bronze"], Path(s.results_dir) / "stream.json", meta)
    trig = {"processingTime": s.trigger}
    queries = [
        ev.writeStream.queryName("bronze").foreachBatch(sink)
          .option("checkpointLocation", str(ck / "bronze")).trigger(**trig).start(),
        job_popularity(ev, s.window, s.watermark).writeStream.queryName("job_popularity")
          .format("parquet").outputMode("append").option("path", paths["job_popularity"])
          .option("checkpointLocation", str(ck / "job_popularity")).trigger(**trig).start(),
        user_activity(ev, s.window, s.watermark).writeStream.queryName("user_activity")
          .format("parquet").outputMode("append").option("path", paths["user_activity"])
          .option("checkpointLocation", str(ck / "user_activity")).trigger(**trig).start(),
    ]
    ready = Path(s.results_dir) / "stream.ready"
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text("ok")                                   # scripts/stream_bench.sh waits for this
    log.info("streaming started (%s); stops after %ss with no new data%s", source, idle_s,
             f" or {expect:,} events" if expect else "")

    t_start = last_change = time.time()
    last_rows = -1
    try:
        while True:
            time.sleep(2)
            for q in queries:
                if q.exception() is not None:
                    raise RuntimeError(f"query {q.name} failed: {q.exception()}")
            busy = any(q.status["isDataAvailable"] or q.status["isTriggerActive"] for q in queries)
            if sink.rows != last_rows or busy:
                last_rows, last_change = sink.rows, time.time()
            # after the last event, give the no-data batch time to advance the watermark and
            # flush the closed windows before stopping
            quiet = time.time() - last_change
            done = expect is not None and sink.rows >= expect and quiet > min(idle_s, 15)
            idle = sink.rows > 0 and quiet > idle_s
            if done or idle or time.time() - t_start > max_wait_s:
                break
    finally:
        for q in queries:
            q.stop()
        ready.unlink(missing_ok=True)
    summary = sink.summary()
    sink.flush()
    log.info("stream done: %s events, lag %s", f"{summary['events']:,}", summary["lag_ms"])
    spark.stop()
    return summary
