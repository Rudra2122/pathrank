"""Write feature snapshots (offline store), register them in Feast, materialize to online.

Snapshots:
  serving cutoff (test): every user (or featurestore.max_users) + every open job -> online store
  earlier cutoffs (ctx, val): a sample of users who apply in the following window, so
  point-in-time retrieval can be demonstrated and tested against AsOf.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..features.pipeline import AsOf
from ..utils import log
from .repo import definitions, get_store, repo_dir, write_yaml

NAN_SENTINEL = -1.0   # Feast/Parquet round-trips NaN poorly for float features; -1 = "never applied"


def _utc(t: pd.Timestamp) -> pd.Timestamp:
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def user_rows(ctx: AsOf, uidx: np.ndarray) -> pd.DataFrame:
    f = ctx.user_features(uidx)
    last = ctx.user_last[uidx]
    return pd.DataFrame({
        "uidx": uidx.astype(np.int64), "event_timestamp": _utc(ctx.t),
        "tok": list(f["tok"].astype(np.int64)), "text": list(f["text"].astype(np.float32)),
        "num": list(f["num"].astype(np.float32)),
        "n_apps": ctx.user_n_apps[uidx].astype(np.float32),
        "days_since_last": np.where(np.isnan(last), NAN_SENTINEL, last).astype(np.float32)})


def job_rows(ctx: AsOf, jidx: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"jidx": jidx.astype(np.int64), "event_timestamp": _utc(ctx.t),
                         "pop_7d": ctx.job_pop_7d[jidx], "pop_all": ctx.job_pop_all[jidx],
                         "age_days": ctx.job_age[jidx], "days_left": ctx.job_left[jidx]})


def build_feature_store(store, cfg) -> dict:
    fc = cfg.featurestore
    c = store.cutoffs
    hb = cfg.two_tower.hash_buckets
    d = repo_dir(cfg)
    rng = np.random.default_rng(cfg.seed)
    serve = AsOf(store, c["test"], c["end"] + pd.Timedelta(seconds=1), hb)
    all_users = np.arange(len(store.users))
    if fc.max_users and len(all_users) > fc.max_users:
        # keep users with history first: they are the ones a notification/recs call hits
        active = np.where(serve.user_n_apps > 0)[0]
        rest = np.setdiff1d(all_users, active)
        all_users = np.sort(np.concatenate([active, rng.choice(rest, max(0, fc.max_users - len(active)), replace=False)])[: fc.max_users])

    user_path, job_path = d / "user_snapshots.parquet", d / "job_snapshots.parquet"
    writer = None
    n_user_rows = 0
    snapshots = [(serve, all_users)]
    for name in ("ctx", "val"):
        nxt = {"ctx": "val", "val": "test"}[name]
        ctx = AsOf(store, c[name], c[nxt], hb)
        cand = np.unique(ctx.labels.uidx)
        snapshots.append((ctx, np.sort(rng.choice(cand, min(fc.pit_users, len(cand)), replace=False))))
    for ctx, users in snapshots:
        for s in range(0, len(users), fc.chunk):
            df = user_rows(ctx, users[s:s + fc.chunk])
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(user_path, tbl.schema)
            writer.write_table(tbl.cast(writer.schema))
            n_user_rows += len(df)
        log.info("user snapshot %s: %s users", ctx.t, f"{len(users):,}")
    writer.close()

    jobs = [job_rows(serve, serve.catalog)]
    for name, nxt in (("ctx", "val"), ("val", "test")):
        ctx = AsOf(store, c[name], c[nxt], hb)
        jobs.append(job_rows(ctx, ctx.catalog))
    jdf = pd.concat(jobs, ignore_index=True)
    jdf.to_parquet(job_path, index=False)

    write_yaml(cfg)
    fs = get_store(cfg)
    fs.apply(definitions(cfg, int(store.manifest["text_dim"])))
    t = _utc(c["test"])
    # online store gets exactly the serving snapshot (rows stamped at the test cutoff)
    fs.materialize(start_date=(t - timedelta(seconds=1)).to_pydatetime(),
                   end_date=(t + timedelta(seconds=1)).to_pydatetime())
    info = {"user_snapshot_rows": int(n_user_rows), "job_snapshot_rows": int(len(jdf)),
            "online_users": int(len(all_users)), "online_jobs": int(len(serve.catalog)),
            "online_store": fc.online, "serving_cutoff": str(c["test"]),
            "user_snapshot_mb": round(user_path.stat().st_size / 2**20, 1),
            "job_snapshot_mb": round(job_path.stat().st_size / 2**20, 1)}
    if fc.online == "sqlite":
        info["online_db_mb"] = round((d / "online.db").stat().st_size / 2**20, 1)
    return info


def historical_user_features(cfg, uidx: np.ndarray, t: pd.Timestamp) -> pd.DataFrame:
    """Point-in-time join through Feast's offline store (what a training job would call)."""
    fs = get_store(cfg)
    ent = pd.DataFrame({"uidx": uidx.astype(np.int64), "event_timestamp": _utc(t)})
    df = fs.get_historical_features(entity_df=ent, features=[f"user_dynamic:{f}" for f in
                                                             ("tok", "text", "num", "n_apps", "days_since_last")]).to_df()
    return df.sort_values("uidx").reset_index(drop=True)
