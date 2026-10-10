"""Stage 3: exposure-aware re-ranking.

Item-side group: jobs in "metro" cities (the cities that together hold `metro_share` of
the open catalog) vs everything else. Small-town jobs tend to get little exposure because
popularity signals favor big markets; Handshake's version of this is the employer at a
school recruiters don't visit.

Re-rank: for each user, take the ranker's top-50 and build the top-10 greedily, guaranteeing
that every prefix of length p contains at least floor(min_share * p) non-metro jobs when the
top-50 has them (FA*IR-style prefix constraint, Zehlike et al. 2017). Sweep min_share and
report the NDCG cost of each level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def metro_flags(store, catalog: np.ndarray, share: float) -> np.ndarray:
    """Boolean per job index: True if the job's city is a metro city in this catalog."""
    jobs = store.jobs.iloc[catalog]
    city = (jobs.state + "|" + jobs.city)
    counts = city.value_counts()
    cum = counts.cumsum() / counts.sum()
    metro = set(cum.index[cum <= share]) | {cum.index[0]}
    flag = np.zeros(len(store.jobs), dtype=bool)
    flag[catalog] = city.isin(metro).to_numpy()
    return flag


def rerank(ranked: pd.DataFrame, protected: np.ndarray, min_share: float, k: int = 10,
           pool: int = 50) -> pd.DataFrame:
    """ranked must be sorted by (uidx, score desc). protected: bool per jidx."""
    out = []
    for u, g in ranked.groupby("uidx", sort=False):
        g = g.head(pool)
        j = g.jidx.to_numpy()
        prot = protected[j]
        chosen, used = [], np.zeros(len(j), bool)
        n_prot = 0
        for p in range(1, min(k, len(j)) + 1):
            need = int(np.floor(min_share * p))
            if n_prot < need and (prot & ~used).any():
                i = int(np.argmax(prot & ~used))
            else:
                i = int(np.argmax(~used))
            used[i] = True
            chosen.append(i)
            n_prot += int(prot[i])
        sel = g.iloc[chosen].copy()
        sel["pos"] = np.arange(len(chosen))
        out.append(sel)
    return pd.concat(out, ignore_index=True)
