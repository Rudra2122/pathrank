"""Versioned feature store + as-of contexts.

build_store():  raw -> data/processed/<source>/<version>/{users,jobs,apps}.parquet,
                job_text.npy, job_title_tok.npy, text_encoder.pkl, manifest.json
AsOf:           everything the models need about the world at cutoff t, computed only from
                data before t. The SAME class is used offline (training/eval) and online
                (serving), which is what keeps training and serving features consistent.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import hash_cat, hash_tokens, log, save_json, load_json
from .text import TextEncoder, job_text, user_text

DEGREE_ORDER = {"None": 0, "High School": 1, "Vocational": 2, "Associate's": 3, "Bachelor's": 4,
                "Master's": 5, "PhD": 6}
TITLE_LEN = 12
HIST_LEN = 48


def index_entities(users: pd.DataFrame, jobs: pd.DataFrame, apps: pd.DataFrame):
    """Dense uidx/jidx ids, apps joined to them and sorted by time (shared with streaming/parity.py)."""
    users = users.reset_index(drop=True)
    users["uidx"] = np.arange(len(users), dtype=np.int64)
    users["degree_ord"] = users["degree"].map(DEGREE_ORDER).fillna(0).astype("int8")
    jobs = jobs.reset_index(drop=True)
    jobs["jidx"] = np.arange(len(jobs), dtype=np.int64)
    apps = apps.merge(users[["user_id", "uidx"]], on="user_id").merge(jobs[["job_id", "jidx"]], on="job_id")
    apps = apps.sort_values("ts").reset_index(drop=True)
    return users, jobs, apps


def time_cutoffs(apps: pd.DataFrame, split) -> dict[str, pd.Timestamp]:
    """Time cutoffs from quantiles of the application timeline."""
    q = apps["ts"].quantile([0.0, split.ctx, split.val, split.test, 1.0])
    return {"start": q.iloc[0], "ctx": q.iloc[1].floor("D"), "val": q.iloc[2].floor("D"),
            "test": q.iloc[3].floor("D"), "end": q.iloc[4]}


def build_store(cfg) -> Path:
    out = cfg.store_dir
    if (out / "manifest.json").exists():
        log.info("feature store %s already built; reusing", out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(cfg.data.raw_dir) / cfg.data.source
    if cfg.data.source == "xing":
        from ..data.xing import load_raw as load_xing
        d = load_xing(str(raw_dir), tuple(cfg.data.xing_types), cfg.data.max_users)
    elif cfg.data.source == "olx":
        from ..data.olx import EVENTS, load_raw as load_olx
        d = load_olx(str(raw_dir), tuple(cfg.data.olx_events or EVENTS), cfg.data.max_users, cfg.seed)
    else:   # careerbuilder and synthetic share the CareerBuilder TSV schema
        from ..data.careerbuilder import load_raw
        d = load_raw(str(raw_dir), cfg.data.desc_chars, cfg.data.max_users)
    users, jobs, apps = index_entities(d["users"], d["jobs"], d["apps"])
    cut = time_cutoffs(apps, cfg.split)

    enc = TextEncoder(cfg.text.backend, cfg.text.st_model, cfg.text.tfidf_dim, cfg.seed)
    jt = job_text(jobs)
    enc.fit(jt, cfg.text.tfidf_fit_rows)
    log.info("encoding %s job texts with %s backend (dim=%d)", f"{len(jt):,}", enc.backend, enc.dim)
    job_vec = enc.encode(jt)
    enc.save(out / "text_encoder.pkl")
    np.save(out / "job_text.npy", job_vec)
    np.save(out / "job_title_tok.npy", hash_tokens(jobs["title"].tolist(), cfg.two_tower.hash_buckets, TITLE_LEN))

    users.to_parquet(out / "users.parquet", index=False)
    jobs.to_parquet(out / "jobs.parquet", index=False)
    apps[["uidx", "jidx", "ts"]].to_parquet(out / "apps.parquet", index=False)

    content_hash = hashlib.sha1(
        json.dumps([len(users), len(jobs), len(apps), str(cut["end"]), cfg.data.max_users,
                    enc.backend]).encode()).hexdigest()[:12]
    manifest = {"version": cfg.feature_version, "source": cfg.data.source, "content_hash": content_hash,
                "n_users": len(users), "n_jobs": len(jobs), "n_apps": len(apps),
                "text_backend": enc.backend, "text_dim": int(enc.dim),
                "cutoffs": {k: str(v) for k, v in cut.items()},
                "apps_per_split": {
                    "retriever_train": int(((apps.ts >= cut["ctx"]) & (apps.ts < cut["val"])).sum()),
                    "ranker_train": int(((apps.ts >= cut["val"]) & (apps.ts < cut["test"])).sum()),
                    "test": int((apps.ts >= cut["test"]).sum())}}
    save_json(manifest, out / "manifest.json")
    log.info("feature store written: %s", manifest)
    return out


@dataclass
class Store:
    users: pd.DataFrame
    jobs: pd.DataFrame
    apps: pd.DataFrame
    job_vec: np.ndarray
    job_title_tok: np.ndarray
    encoder: TextEncoder
    manifest: dict
    # static per-entity categorical ids
    job_state: np.ndarray = None
    job_city: np.ndarray = None
    user_state: np.ndarray = None
    user_city: np.ndarray = None
    user_major: np.ndarray = None
    user_key: np.ndarray = None   # "STATE|city" strings, computed once
    job_key: np.ndarray = None
    np: dict = None

    @property
    def cutoffs(self) -> dict[str, pd.Timestamp]:
        return {k: pd.Timestamp(v) for k, v in self.manifest["cutoffs"].items()}


def load_store(cfg) -> Store:
    p = cfg.store_dir
    s = Store(users=pd.read_parquet(p / "users.parquet"), jobs=pd.read_parquet(p / "jobs.parquet"),
              apps=pd.read_parquet(p / "apps.parquet"), job_vec=np.load(p / "job_text.npy"),
              job_title_tok=np.load(p / "job_title_tok.npy"), encoder=TextEncoder.load(p / "text_encoder.pkl"),
              manifest=load_json(p / "manifest.json"))
    B = 4096
    s.job_state = hash_cat(s.jobs["state"].tolist(), 256)
    s.user_state = hash_cat(s.users["state"].tolist(), 256)
    s.job_city = hash_cat((s.jobs["state"] + "|" + s.jobs["city"]).tolist(), B * 8)
    s.user_city = hash_cat((s.users["state"] + "|" + s.users["city"]).tolist(), B * 8)
    s.user_major = hash_cat(s.users["major"].tolist(), B)
    # Location strings -> integer codes shared by users and jobs (cheap compares, little memory).
    nu = len(s.users)

    def codes(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        c, _ = pd.factorize(pd.concat([a, b], ignore_index=True))
        return c[:nu].astype(np.int32), c[nu:].astype(np.int32)

    s.user_key, s.job_key = codes(s.users["state"] + "|" + s.users["city"], s.jobs["state"] + "|" + s.jobs["city"])
    s.np = {}
    for col in ("state", "zip3"):
        s.np[f"u_{col}"], s.np[f"j_{col}"] = codes(s.users[col], s.jobs[col])
    s.np["u_has_city"] = (s.users["city"] != "").to_numpy()
    s.np["u_has_zip3"] = (s.users["zip3"] != "").to_numpy()
    s.np["j_title"] = s.jobs["title"].to_numpy(object)
    return s


INFERRED_DATE_SOURCES = ("xing", "olx")   # no posting dates: windows were inferred from interactions


def job_window(store: Store, t: pd.Timestamp, horizon: pd.Timestamp,
               active_days: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(catalog jidx, job_age days, job_left days) at cutoff t, using only what is knowable at t.

    With real posting windows (CareerBuilder, synthetic): open at some point in [t, horizon).
    With inferred windows (XING, OLX) the end date is "last interaction + 14d", which peeks past t.
    There a job is in the catalog if it had an interaction in [t - active_days, t) or is newly
    posted in [t, horizon) (a real system knows its postings when they go live), and job_left is
    unknown (NaN) instead of a leaked countdown.
    """
    j = store.jobs
    age = ((t - j.start_date).dt.total_seconds() / 86400).to_numpy(np.float32)
    if store.manifest.get("source") not in INFERRED_DATE_SOURCES:
        cat = j.index[(j.start_date < horizon) & (j.end_date >= t)].to_numpy()
        left = ((j.end_date - t).dt.total_seconds() / 86400).to_numpy(np.float32)
        return cat, age, left
    a = store.apps
    h = a[(a.ts < t) & (a.ts >= t - pd.Timedelta(days=active_days))]
    active = np.zeros(len(j), dtype=bool)
    active[h.jidx.to_numpy()] = True
    start = j.start_date.to_numpy()
    new = (start >= np.datetime64(t - pd.Timedelta(days=1))) & (start < np.datetime64(horizon))
    cat = np.flatnonzero(active | new)
    return cat, age, np.full(len(j), np.nan, np.float32)


class AsOf:
    """World state at cutoff t (strictly before t), plus the label window [t, t_end)."""

    def __init__(self, store: Store, t: pd.Timestamp, t_end: pd.Timestamp | None = None,
                 hash_buckets: int = 2 ** 18, catalog_days: int = 7):
        self.s, self.t, self.t_end = store, t, t_end
        a = store.apps
        self.hist = a[a.ts < t]
        self.labels = a[(a.ts >= t) & (a.ts < t_end)] if t_end is not None else a.iloc[:0]
        self.hash_buckets = hash_buckets

        # catalog: jobs open at some point in [t, horizon); horizon = label window end if known
        horizon = t_end if t_end is not None else t + pd.Timedelta(days=catalog_days)
        j = store.jobs
        self.catalog, self.job_age, self.job_left = job_window(store, t, horizon)
        self.catalog_mask = np.zeros(len(j), dtype=bool)
        self.catalog_mask[self.catalog] = True

        # job popularity as of t
        nJ = len(j)
        self.job_pop_all = np.bincount(self.hist.jidx, minlength=nJ).astype(np.float32)
        recent = self.hist[self.hist.ts >= t - pd.Timedelta(days=7)]
        self.job_pop_7d = np.bincount(recent.jidx, minlength=nJ).astype(np.float32)

        # user activity as of t
        nU = len(store.users)
        g = self.hist.groupby("uidx")["ts"]
        self.user_n_apps = np.zeros(nU, np.float32)
        self.user_last = np.full(nU, np.nan, np.float32)
        cnt = g.size()
        self.user_n_apps[cnt.index.to_numpy()] = cnt.to_numpy()
        last = g.max()
        self.user_last[last.index.to_numpy()] = ((t - last).dt.total_seconds() / 86400).to_numpy()
        hu = self.hist.uidx.to_numpy()
        order = np.lexsort((self.hist.ts.to_numpy(), hu))   # by user, then time
        self._hist_j = self.hist.jidx.to_numpy()[order]
        self._hist_off = np.searchsorted(hu[order], np.arange(nU + 1))
        del order, hu
        self.cache: dict = {}   # per-context caches (FAISS indexes, location tables)

    def history_titles(self, uidx: np.ndarray, last: int = 10) -> list[str]:
        titles = self.s.np["j_title"]
        out = []
        for u in uidx:
            a, b = self._hist_off[u], self._hist_off[u + 1]
            out.append(" | ".join(titles[self._hist_j[max(a, b - last):b]]))
        return out

    # ---- per-user features for a subset of users ------------------------------------
    def user_features(self, uidx: np.ndarray) -> dict[str, np.ndarray]:
        s = self.s
        u = s.users.iloc[uidx]
        applied = pd.Series(self.history_titles(uidx, 10))   # last 10 applied titles before t
        hist_text = (u["work_titles"].fillna("").to_numpy() + " | " + applied.to_numpy())
        tok = hash_tokens(list(hist_text), self.hash_buckets, HIST_LEN)
        prof_text = user_text(u.reset_index(drop=True), applied)
        vec = s.encoder.encode(prof_text)
        num = np.stack([
            np.log1p(u["years_exp"].to_numpy(np.float32)),
            np.log1p(u["work_history_count"].to_numpy(np.float32)),
            u["employed"].to_numpy(np.float32),
            u["managed_others"].to_numpy(np.float32),
            np.log1p(self.user_n_apps[uidx]),
            (self.user_n_apps[uidx] == 0).astype(np.float32),
        ], 1)
        return {"uidx": uidx, "state": s.user_state[uidx], "city": s.user_city[uidx],
                "degree": u["degree_ord"].to_numpy(np.int64), "major": s.user_major[uidx],
                "tok": tok, "text": vec, "num": num.astype(np.float32)}

    def job_features(self, jidx: np.ndarray) -> dict[str, np.ndarray]:
        s = self.s
        return {"jidx": jidx, "state": s.job_state[jidx], "city": s.job_city[jidx],
                "tok": s.job_title_tok[jidx], "text": s.job_vec[jidx]}

    def label_sets(self, uidx: np.ndarray) -> dict[int, set]:
        lab = self.labels[self.labels.uidx.isin(uidx)]
        return lab.groupby("uidx")["jidx"].agg(set).to_dict()

    def eval_users(self, n: int | None, seed: int, require_catalog_label: bool = True) -> np.ndarray:
        lab = self.labels
        if require_catalog_label:
            lab = lab[self.catalog_mask[lab.jidx.to_numpy()]]
        users = np.sort(lab.uidx.unique())
        if n and len(users) > n:
            users = np.sort(np.random.default_rng(seed).choice(users, n, replace=False))
        return users
