"""Parity checks and the Spark-vs-pandas timing.

batch_parity()   Spark batch as-of features vs the pandas AsOf class (the code training and serving
                 use), same events, same cutoff. Expected max |difference| = 0 for all four features.
stream_parity()  streaming windows in the lake vs the same windows from a Spark batch job, plus
                 Kafka -> S3 completeness (rows landed vs produced, duplicate (partition, offset)).
"""
from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.interactions import to_schema
from ..data.olx import EVENTS
from ..features.pipeline import AsOf, Store, index_entities, time_cutoffs
from ..utils import load_json, log
from .batch import asof_features, read_events, windows
from .spark import lake, session
from .stream import job_popularity, user_activity  # noqa: F401  (same functions the stream ran)


def _machine() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count()}


def pandas_reference(cfg, events_path: str, reference: str = "rebuild"):
    """-> (users, jobs, apps, t, timings). reference=rebuild: events Parquet -> to_schema (the OLX loader
    path); reference=store: the feature store already on disk (cfg.store_dir) and its test cutoff."""
    tm: dict[str, float] = {}
    t0 = time.perf_counter()
    if reference == "store":
        p = cfg.store_dir
        users = pd.read_parquet(p / "users.parquet", columns=["user_id"])
        jobs = pd.read_parquet(p / "jobs.parquet", columns=["job_id", "start_date", "end_date"])
        apps = pd.read_parquet(p / "apps.parquet")
        t = pd.Timestamp(load_json(p / "manifest.json")["cutoffs"]["test"])
        tm["read_s"] = time.perf_counter() - t0
    else:
        import pyarrow.parquet as pq
        d = pq.read_table(events_path, columns=["user", "item", "event", "timestamp"],
                          read_dictionary=["event"]).to_pandas()
        tm["read_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        d = d[d.event.isin(list(cfg.data.olx_events or EVENTS))]
        raw = pd.DataFrame({"user_id": d.user.to_numpy(), "job_id": d.item.to_numpy(),
                            "ts": pd.to_datetime(d.timestamp.to_numpy(), unit="s")})
        del d
        s = to_schema(raw, cfg.data.max_users, min_user_apps=2, seed=cfg.seed)
        del raw
        users, jobs, apps = index_entities(s["users"], s["jobs"], s["apps"])
        apps = apps[["uidx", "jidx", "ts"]]
        t = time_cutoffs(apps, cfg.split)["test"]
        tm["schema_s"] = time.perf_counter() - t1
    return users, jobs, apps, t, tm


def batch_parity(cfg, events_path: str, reference: str = "rebuild") -> dict:
    s = cfg.streaming
    users, jobs, apps, t, tm = pandas_reference(cfg, events_path, reference)
    t2 = time.perf_counter()
    store = Store(users=users, jobs=jobs, apps=apps, job_vec=None, job_title_tok=None, encoder=None,
                  manifest={"source": "olx"})
    a = AsOf(store, t)
    tm["asof_s"] = time.perf_counter() - t2
    tm["total_s"] = sum(tm.values())
    ref_j = pd.DataFrame({"item": jobs.job_id.to_numpy(), "pop_all": a.job_pop_all, "pop_7d": a.job_pop_7d})
    ref_u = pd.DataFrame({"user": users.user_id.to_numpy(), "n_apps": a.user_n_apps, "last_days": a.user_last})
    n_pairs = len(apps)
    t_epoch = int(t.value // 10**9)
    log.info("pandas: %s pairs, %s users, %s jobs, cutoff %s, %s", f"{n_pairs:,}", f"{len(users):,}",
             f"{len(jobs):,}", t, {k: round(v, 1) for k, v in tm.items()})
    restrict = cfg.data.max_users is not None or reference == "store"
    del a, store, apps

    with tempfile.TemporaryDirectory() as tmp:
        users_path = None
        if restrict:
            users_path = str(Path(tmp) / "users.parquet")
            pd.DataFrame({"user": users.user_id.to_numpy()}).to_parquet(users_path, index=False)
        del users, jobs
        spark = session(cfg, "pathrank-batch-parity", shuffle=s.batch_shuffle_partitions)
        out_j, out_u = lake(cfg, "batch", "asof_jobs"), lake(cfg, "batch", "asof_users")
        t3 = time.perf_counter()
        sp_apps, sp_j, sp_u = asof_features(spark, events_path, cfg.data.olx_events or EVENTS, t_epoch, users_path)
        sp_j.write.mode("overwrite").parquet(out_j)
        sp_u.write.mode("overwrite").parquet(out_u)
        spark_s = time.perf_counter() - t3
        sp_pairs = sp_apps.count()
        got_j = spark.read.parquet(out_j).toPandas()
        got_u = spark.read.parquet(out_u).toPandas()
        spark_meta = {"version": spark.version, "master": spark.sparkContext.master,
                      "cores": spark.sparkContext.defaultParallelism}
        spark.stop()

    def compare(ref, got, key, cols):
        m = ref.merge(got, on=key, how="outer", suffixes=("_pd", "_sp"), indicator=True)
        res = {"only_pandas": int((m._merge == "left_only").sum()), "only_spark": int((m._merge == "right_only").sum())}
        both = m[m._merge == "both"]
        for c in cols:
            x = both[f"{c}_pd"].to_numpy(np.float64)
            y = both[f"{c}_sp"].astype("float64").to_numpy().astype(np.float32).astype(np.float64)
            nan_x, nan_y = np.isnan(x), np.isnan(y)
            res[f"{c}_nan_mismatch"] = int((nan_x != nan_y).sum())
            ok = ~(nan_x | nan_y)
            res[f"{c}_max_abs_diff"] = float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0
        return res

    res_j = compare(ref_j, got_j, "item", ["pop_all", "pop_7d"])
    res_u = compare(ref_u, got_u, "user", ["n_apps", "last_days"])
    max_diff = max(v for r in (res_j, res_u) for k, v in r.items() if k.endswith("max_abs_diff"))
    mismatches = sum(v for r in (res_j, res_u) for k, v in r.items() if not k.endswith("max_abs_diff"))
    out = {"events_parquet": events_path, "reference": reference, "cutoff": str(t),
           "pairs": n_pairs, "pairs_spark": int(sp_pairs), "users": len(ref_u), "jobs": len(ref_j),
           "jobs_check": res_j, "users_check": res_u, "max_abs_diff": max_diff,
           "row_or_nan_mismatches": int(mismatches), "pass": max_diff == 0 and mismatches == 0
           and n_pairs == sp_pairs,
           "pandas_s": {k: round(v, 2) for k, v in tm.items()}, "spark_s": round(spark_s, 2),
           "speedup": round(tm["total_s"] / spark_s, 2) if spark_s else None,
           "spark": spark_meta, "machine": _machine()}
    _save(cfg, "batch_parity.json", out)
    log.info("batch parity: max|diff|=%s, mismatches=%d, pandas %.1fs vs spark %.1fs -> %s",
             max_diff, mismatches, tm["total_s"], spark_s, "PASS" if out["pass"] else "FAIL")
    return out


def stream_parity(cfg, events_path: str) -> dict:
    """Stream output vs batch recompute of the same windows; Kafka -> S3 completeness."""
    from pyspark.sql import functions as F

    s = cfg.streaming
    spark = session(cfg, "pathrank-stream-parity", shuffle=s.batch_shuffle_partitions)
    ev = read_events(spark, events_path)
    n_events = ev.count()
    max_ts = ev.agg(F.max("ts")).first()[0]
    # windows that can still be open when the stream stops: end > max event time - watermark
    closed_by = int(max_ts) - _interval_seconds(s.watermark)

    b_job, b_user = windows(spark, events_path, s.window)
    res = {"events_parquet": events_path, "events_in_parquet": int(n_events), "window": s.window,
           "watermark": s.watermark, "closed_by": str(pd.Timestamp(closed_by, unit="s"))}
    for name, b, key, cols in [("job_popularity", b_job, "item", ["events", "contacts"]),
                               ("user_activity", b_user, "user", ["events", "contacts", "last_ts"])]:
        st = spark.read.parquet(lake(cfg, "features", f"{name}_1h"))
        j = b.alias("b").join(st.alias("s"), ["window_start", key], "full_outer")
        both = F.col("b.events").isNotNull() & F.col("s.events").isNotNull()
        b_only = F.col("s.events").isNull()
        aggs = [F.sum(both.cast("long")).alias("matched"),
                F.sum(F.col("b.events").isNull().cast("long")).alias("stream_only"),
                F.sum((b_only & (F.unix_timestamp("b.window_end") > F.lit(closed_by))).cast("long")).alias("open_at_end"),
                F.sum((b_only & (F.unix_timestamp("b.window_end") <= F.lit(closed_by))).cast("long")).alias("missing_closed"),
                F.sum(F.when(b_only, F.col("b.events")).otherwise(0)).alias("events_in_open_windows"),
                F.count(F.lit(1)).alias("batch_or_stream_windows")]
        aggs += [F.max(F.when(both, F.abs(F.col(f"b.{c}") - F.col(f"s.{c}")))).alias(f"{c}_max_abs_diff")
                 for c in cols]
        r = j.agg(*aggs).first().asDict()
        r = {k: (int(v) if v is not None else 0) for k, v in r.items()}
        res[name] = r
        log.info("%s: %s", name, r)

    bronze = lake(cfg, "bronze", "job_events")
    try:
        br = spark.read.parquet(bronze)
        landed = br.count()
        uniq = br.select("partition", "offset").distinct().count()
        res["bronze"] = {"rows": int(landed), "duplicates": int(landed - uniq)}
    except Exception as e:  # noqa: BLE001  (no bronze yet)
        res["bronze"] = {"error": str(e)[:200]}
    prod = Path(s.results_dir) / "producer_lag.json"     # the producer run the stream consumed
    if prod.exists():
        res["producer_acked"] = int(json.loads(prod.read_text()).get("acked", 0))
    spark.stop()

    diffs = [res[n][k] for n in ("job_popularity", "user_activity") for k in res[n] if k.endswith("max_abs_diff")]
    landed_ok = res["bronze"].get("rows") == res.get("producer_acked", n_events) and \
        res["bronze"].get("duplicates") == 0
    res["max_abs_diff"] = max(diffs) if diffs else None
    res["pass"] = bool(diffs) and max(diffs) == 0 and landed_ok and all(
        res[n]["stream_only"] == 0 and res[n]["missing_closed"] == 0 for n in ("job_popularity", "user_activity"))
    _save(cfg, "stream_parity.json", res)
    log.info("stream parity: %s", "PASS" if res["pass"] else "FAIL")
    return res


def _interval_seconds(text: str) -> int:
    n, unit = text.split()
    mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
    return int(float(n) * mult[unit.rstrip("s")])


def _save(cfg, name: str, obj: dict) -> None:
    p = Path(cfg.streaming.results_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / name).write_text(json.dumps(obj, indent=1, default=str))
