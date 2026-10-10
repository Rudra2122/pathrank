"""Ranking metrics. Every eval user counts, including users whose positives were never retrieved."""
from __future__ import annotations

import numpy as np
import pandas as pd

_DISC = 1.0 / np.log2(np.arange(2, 202))


def per_user_metrics(ranked: pd.DataFrame, positives: dict[int, set], users: np.ndarray,
                     score_col: str = "score", ks=(10, 100)) -> pd.DataFrame:
    """ranked: rows (uidx, jidx, score_col); positives: uidx -> set(jidx)."""
    r = ranked.sort_values(["uidx", score_col], ascending=[True, False], kind="stable")
    lists = r.groupby("uidx")["jidx"].apply(lambda x: x.to_numpy()).to_dict()
    rows = []
    for u in users:
        pos = positives.get(u, set())
        lst = lists.get(u, np.array([], dtype=np.int64))
        rel = np.fromiter((j in pos for j in lst[: max(ks)]), dtype=bool, count=min(len(lst), max(ks)))
        n = len(pos)
        row = {"uidx": u, "n_pos": n}
        r10 = rel[:10]
        dcg = (r10 * _DISC[: len(r10)]).sum()
        idcg = _DISC[: min(n, 10)].sum()
        row["ndcg@10"] = dcg / idcg if idcg > 0 else 0.0
        hits = np.cumsum(r10)
        prec = hits / np.arange(1, len(r10) + 1)
        row["map@10"] = (prec * r10).sum() / min(n, 10) if n else 0.0
        row["hit@10"] = float(r10.any())
        for k in ks:
            row[f"recall@{k}"] = rel[:k].sum() / n if n else 0.0
        row["cand_recall"] = (np.isin(lst, list(pos)).sum() / n) if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(pu: pd.DataFrame) -> dict:
    cols = [c for c in pu.columns if c not in ("uidx", "n_pos")]
    return {c: float(pu[c].mean()) for c in cols} | {"n_users": int(len(pu))}


def exposure(ranked_topk: pd.DataFrame, group_col: str, k: int = 10) -> pd.Series:
    """Position-weighted exposure share per group over everyone's top-k."""
    r = ranked_topk.copy()
    r["pos"] = r.groupby("uidx").cumcount()
    r = r[r.pos < k]
    r["w"] = 1.0 / np.log2(r.pos + 2)
    e = r.groupby(group_col)["w"].sum()
    return e / e.sum()
