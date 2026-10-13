"""Spark batch jobs over the full events Parquet.

asof_features()  the four pandas AsOf features (job_pop_all, job_pop_7d, user_n_apps, user_last) at
                 a cutoff t, recomputed in Spark from raw events. Mirrors data/olx.py + to_schema:
                 configured event types -> first touch per (user, job) -> users with >= 2 jobs.
windows()        the streaming job's windowed features, same functions, run as a batch job.
"""
from __future__ import annotations

from .stream import job_popularity, user_activity


def read_events(spark, path: str):
    from pyspark.sql import functions as F
    return (spark.read.parquet(path).withColumnRenamed("timestamp", "ts")
            .withColumn("event_time", F.timestamp_seconds("ts")))


def asof_features(spark, events_path: str, events, t: int, users_path: str | None = None):
    """t: cutoff in unix seconds. users_path: restrict to these users (the pandas side subsampled)."""
    from pyspark import StorageLevel
    from pyspark.sql import functions as F

    ev = spark.read.parquet(events_path).where(F.col("event").isin(list(events)))
    pairs = ev.groupBy("user", "item").agg(F.min("timestamp").alias("ts"))      # first touch
    keep = pairs.groupBy("user").count().where(F.col("count") >= 2).select("user")
    if users_path:
        keep = keep.join(spark.read.parquet(users_path), "user", "left_semi")
    apps = pairs.join(keep, "user", "left_semi").persist(StorageLevel.MEMORY_AND_DISK)

    hist = apps.where(F.col("ts") < F.lit(t))
    recent = (F.col("ts") >= F.lit(t - 7 * 86400)).cast("long")
    jobs = (apps.select("item").distinct()
            .join(hist.groupBy("item").agg(F.count(F.lit(1)).alias("pop_all"), F.sum(recent).alias("pop_7d")),
                  "item", "left")
            .fillna(0, ["pop_all", "pop_7d"]))
    users = (apps.select("user").distinct()
             .join(hist.groupBy("user").agg(F.count(F.lit(1)).alias("n_apps"), F.max("ts").alias("last_ts")),
                   "user", "left")
             .fillna(0, ["n_apps"])
             .withColumn("last_days", (F.lit(t) - F.col("last_ts")) / F.lit(86400.0))
             .drop("last_ts"))
    return apps, jobs, users


def windows(spark, events_path: str, window: str):
    ev = read_events(spark, events_path)
    return job_popularity(ev, window), user_activity(ev, window)
