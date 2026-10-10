"""Non-learned retrievers: text match (cold-start), location popularity, global popularity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.pipeline import AsOf
from .index import CatalogIndex


def text_retrieve(ctx: AsOf, user_text_vecs: np.ndarray, k: int):
    if "txt_index" not in ctx.cache:
        ctx.cache["txt_index"] = CatalogIndex(ctx.s.job_vec[ctx.catalog], ctx.catalog)
    return ctx.cache["txt_index"].search(user_text_vecs, k)


def _recent_score(ctx: AsOf, jidx: np.ndarray) -> np.ndarray:
    # recent popularity, ties broken by freshness (new postings have no history yet)
    return ctx.job_pop_7d[jidx] + 0.1 * np.log1p(ctx.job_pop_all[jidx]) - 0.001 * np.clip(ctx.job_age[jidx], 0, None)


def popularity(ctx: AsOf, n_users: int, k: int):
    cat = ctx.catalog
    sc = _recent_score(ctx, cat)
    top = cat[np.argsort(-sc, kind="stable")[:k]]
    return (np.tile(sc[np.argsort(-sc, kind="stable")[:k]], (n_users, 1)).astype(np.float32),
            np.tile(top, (n_users, 1)))


def location_popularity(ctx: AsOf, uidx: np.ndarray, k: int):
    """Top recent jobs in the user's city, then state, then nationwide."""
    s = ctx.s
    key = ("lp", k)
    if key not in ctx.cache:
        cat = ctx.catalog
        sc = _recent_score(ctx, cat)
        df = pd.DataFrame({"jidx": cat, "sc": sc, "city": s.job_key[cat],
                           "state": s.np["j_state"][cat]}).sort_values("sc", ascending=False, kind="stable")
        df["r"] = df.groupby("city").cumcount()
        by_city = df[df.r < k].groupby("city")["jidx"].agg(list).to_dict()
        df["r"] = df.groupby("state").cumcount()
        by_state = df[df.r < k].groupby("state")["jidx"].agg(list).to_dict()
        ctx.cache[key] = (by_city, by_state, df["jidx"].to_numpy()[:k])
    by_city, by_state, glob = ctx.cache[key]
    ucity = s.user_key[uidx]
    ustate = s.np["u_state"][uidx]
    J = np.full((len(uidx), k), -1, np.int64)
    S = np.full((len(uidx), k), -np.inf, np.float32)
    for r in range(len(uidx)):
        lst, seen = [], set()
        for src in (by_city.get(ucity[r], ()), by_state.get(ustate[r], ()), glob):
            for j in src:
                if j not in seen:
                    seen.add(j)
                    lst.append(j)
                    if len(lst) == k:
                        break
            if len(lst) == k:
                break
        J[r, : len(lst)] = lst
        S[r, : len(lst)] = -np.arange(len(lst), dtype=np.float32)  # rank-based score
    return S, J
