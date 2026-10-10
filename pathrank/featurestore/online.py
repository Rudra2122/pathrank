"""Serving-time context backed by the Feast online store.

OnlineContext exposes the same interface the Recommender uses on AsOf, but:
  * user dynamic features (activity, history tokens, profile embedding) are fetched per
    request from the online store; nothing is recomputed from raw applications
  * job dynamic features for the open catalog are loaded from the online store at startup
    (and on refresh(), which a real deployment calls after each materialization)
  * static attributes come from the versioned tables, as in training.
"""
from __future__ import annotations

import contextlib
import threading
import time

import numpy as np
import pandas as pd

from ..features.pipeline import Store, job_window
from ..utils import log
from .materialize import NAN_SENTINEL
from .repo import JOB_FEATURES, get_store

USER_REFS = [f"user_dynamic:{f}" for f in ("tok", "text", "num", "n_apps", "days_since_last")]
JOB_REFS = [f"job_dynamic:{f}" for f in JOB_FEATURES]


class OnlineContext:
    def __init__(self, store: Store, cfg, t: pd.Timestamp, catalog_days: int = 7):
        self.s, self.cfg, self.t = store, cfg, t
        self.fs = get_store(cfg)
        j = store.jobs
        # catalog rule is metadata (posting windows), identical to AsOf
        end = store.cutoffs["end"] + pd.Timedelta(seconds=1)
        horizon = end if t == store.cutoffs["test"] else t + pd.Timedelta(days=catalog_days)
        self.catalog, self.job_age, self.job_left = job_window(store, t, horizon)
        self.catalog_mask = np.zeros(len(j), dtype=bool)
        self.catalog_mask[self.catalog] = True
        nU, nJ = len(store.users), len(j)
        self.user_n_apps = np.zeros(nU, np.float32)
        self.user_last = np.full(nU, np.nan, np.float32)
        self.job_pop_7d = np.zeros(nJ, np.float32)
        self.job_pop_all = np.zeros(nJ, np.float32)
        self.cache: dict = {}
        self.stats = {"user_fetch_ms": []}
        # Feast's SQLite online store shares one sqlite3 connection across threads; concurrent reads
        # (API threadpool, concurrent agent evals) fail or return empty rows. Serialize them.
        # Redis' client is thread-safe, so Redis reads stay concurrent.
        self._read_lock = threading.Lock() if cfg.featurestore.online == "sqlite" else contextlib.nullcontext()
        self.refresh()

    def refresh(self, batch: int = 20_000) -> None:
        t0 = time.perf_counter()
        cat = self.catalog
        for s in range(0, len(cat), batch):
            rows = [{"jidx": int(x)} for x in cat[s:s + batch]]
            with self._read_lock:
                r = self.fs.get_online_features(features=JOB_REFS, entity_rows=rows).to_dict()
            ids = np.asarray(r["jidx"])
            self.job_pop_7d[ids] = np.nan_to_num(np.asarray(r["pop_7d"], dtype=np.float32))
            self.job_pop_all[ids] = np.nan_to_num(np.asarray(r["pop_all"], dtype=np.float32))
            self.job_age[ids] = np.asarray(r["age_days"], dtype=np.float32)
            # None/NaN (XING/OLX: unknown end dates) -> NaN, same as the offline AsOf value
            self.job_left[ids] = pd.to_numeric(pd.Series(r["days_left"]), errors="coerce").to_numpy(np.float32)
        self.cache.clear()
        log.info("online catalog refresh: %s jobs in %.1fs", f"{len(cat):,}", time.perf_counter() - t0)

    def user_features(self, uidx: np.ndarray) -> dict[str, np.ndarray]:
        t0 = time.perf_counter()
        with self._read_lock:
            r = self.fs.get_online_features(features=USER_REFS,
                                            entity_rows=[{"uidx": int(u)} for u in uidx]).to_dict()
        self.stats["user_fetch_ms"].append((time.perf_counter() - t0) * 1000)
        missing = [int(u) for u, v in zip(uidx, r["n_apps"]) if v is None]
        if missing:
            raise KeyError(f"users not in online store: {missing[:5]}")
        s = self.s
        n_apps = np.asarray(r["n_apps"], np.float32)
        last = np.asarray(r["days_since_last"], np.float32)
        self.user_n_apps[uidx] = n_apps
        self.user_last[uidx] = np.where(last == NAN_SENTINEL, np.nan, last)
        u = s.users.iloc[uidx]
        return {"uidx": uidx, "state": s.user_state[uidx], "city": s.user_city[uidx],
                "degree": u["degree_ord"].to_numpy(np.int64), "major": s.user_major[uidx],
                "tok": np.asarray(r["tok"], dtype=np.int32), "text": np.asarray(r["text"], dtype=np.float32),
                "num": np.asarray(r["num"], dtype=np.float32)}

    def job_features(self, jidx: np.ndarray) -> dict[str, np.ndarray]:
        s = self.s
        return {"jidx": jidx, "state": s.job_state[jidx], "city": s.job_city[jidx],
                "tok": s.job_title_tok[jidx], "text": s.job_vec[jidx]}
