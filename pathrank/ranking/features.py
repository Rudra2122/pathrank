"""Candidate merging (stage 1 output) and ranker features (stage 2 input)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.pipeline import AsOf

SOURCES = ("tt", "txt", "gcn", "lp")

RANKER_V1 = ["tt_score", "tt_rank", "txt_score", "txt_rank", "gcn_score", "gcn_rank", "lp_rank", "n_sources"]
RANKER_V2 = RANKER_V1 + [
    "same_city", "same_state", "same_zip3", "job_pop_7d", "job_pop_all", "job_age_days", "job_days_left",
    "title_overlap", "user_n_apps", "user_days_since_last", "user_years_exp", "user_degree", "user_cold",
    "job_state_share",
]

FEATURE_TEXT = {  # used for human-readable "why this job" reasons
    "tt_score": "matches your profile and past applications",
    "txt_score": "description matches your background",
    "gcn_score": "people with similar applications applied here",
    "lp_rank": "popular near you",
    "same_city": "it's in your city",
    "same_state": "it's in your state",
    "same_zip3": "it's close to your zip code",
    "title_overlap": "title matches roles you've held or applied to",
    "job_pop_7d": "getting a lot of applications this week",
    "job_age_days": "recently posted",
    "job_days_left": "still open for a while",
}


def _long(uidx: np.ndarray, S: np.ndarray, J: np.ndarray, name: str) -> pd.DataFrame:
    k = J.shape[1]
    df = pd.DataFrame({"uidx": np.repeat(uidx, k), "jidx": J.ravel(), f"{name}_score": S.ravel(),
                       f"{name}_rank": np.tile(np.arange(1, k + 1, dtype=np.float32), len(uidx))})
    return df[df.jidx >= 0]


def merge_candidates(uidx: np.ndarray, sources: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    out = None
    for name in SOURCES:
        if name not in sources:
            continue
        S, J = sources[name]
        df = _long(uidx, S, J, name)
        out = df if out is None else out.merge(df, on=["uidx", "jidx"], how="outer")
    for name in SOURCES:
        for c in (f"{name}_score", f"{name}_rank"):
            if c not in out:
                out[c] = np.nan
    out["n_sources"] = out[[f"{n}_rank" for n in SOURCES]].notna().sum(1).astype(np.float32)
    out = out.drop(columns=["lp_score"])
    return out.reset_index(drop=True)


def _title_overlap(utok: np.ndarray, jtok: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    out = np.zeros(len(utok), np.float32)
    for s in range(0, len(utok), chunk):
        u = utok[s:s + chunk]
        j = jtok[s:s + chunk]
        hit = ((j[:, :, None] == u[:, None, :]) & (j[:, :, None] > 0)).any(-1).sum(-1)
        nj = (j > 0).sum(-1)
        out[s:s + chunk] = hit / np.maximum(nj, 1)
    return out


def pair_features(ctx: AsOf, cand: pd.DataFrame, user_feats: dict) -> pd.DataFrame:
    s = ctx.s
    u = cand.uidx.to_numpy()
    j = cand.jidx.to_numpy()
    upos = pd.Series(np.arange(len(user_feats["uidx"])), index=user_feats["uidx"])[u].to_numpy()
    U = s.users
    df = cand.copy()
    N = s.np
    df["same_city"] = ((s.user_key[u] == s.job_key[j]) & N["u_has_city"][u]).astype(np.float32)
    df["same_state"] = (N["u_state"][u] == N["j_state"][j]).astype(np.float32)
    df["same_zip3"] = ((N["u_zip3"][u] == N["j_zip3"][j]) & N["u_has_zip3"][u]).astype(np.float32)
    df["job_pop_7d"] = np.log1p(ctx.job_pop_7d[j])
    df["job_pop_all"] = np.log1p(ctx.job_pop_all[j])
    df["job_age_days"] = ctx.job_age[j]
    df["job_days_left"] = ctx.job_left[j]
    df["title_overlap"] = _title_overlap(user_feats["tok"][upos], s.job_title_tok[j])
    df["user_n_apps"] = np.log1p(ctx.user_n_apps[u])
    df["user_days_since_last"] = ctx.user_last[u]
    df["user_years_exp"] = U.years_exp.to_numpy(np.float32)[u]
    df["user_degree"] = U.degree_ord.to_numpy(np.float32)[u]
    df["user_cold"] = (ctx.user_n_apps[u] == 0).astype(np.float32)
    # share of this user's candidates in the job's state (how "local" this slate is)
    df["job_state_share"] = df.groupby(["uidx", "same_state"])["jidx"].transform("size") / \
        df.groupby("uidx")["jidx"].transform("size")
    return df


def attach_labels(ctx: AsOf, df: pd.DataFrame) -> pd.DataFrame:
    lab = ctx.labels[["uidx", "jidx"]].drop_duplicates().assign(label=1)
    df = df.merge(lab, on=["uidx", "jidx"], how="left")
    df["label"] = df["label"].fillna(0).astype(np.int8)
    return df
