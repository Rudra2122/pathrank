"""SHAP explanations for the ranker: global importance + per-recommendation 'why this job'."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_TEXT


def shap_values(ranker, df: pd.DataFrame, max_rows: int = 20_000, seed: int = 0):
    import shap

    X = df[ranker.features]
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    expl = shap.TreeExplainer(ranker.model)
    vals = expl.shap_values(X)
    return X, np.asarray(vals)


def global_importance(ranker, df: pd.DataFrame) -> pd.Series:
    X, vals = shap_values(ranker, df)
    return pd.Series(np.abs(vals).mean(0), index=X.columns).sort_values(ascending=False)


# features whose contribution only makes sense to show when it pushed the score up
_REASON_FEATURES = [f for f in FEATURE_TEXT]


def reasons(ranker, rows: pd.DataFrame, top: int = 2) -> list[str]:
    """Human-readable reason per row from its largest positive SHAP contributions."""
    import shap

    vals = np.asarray(shap.TreeExplainer(ranker.model).shap_values(rows[ranker.features]))
    names = ranker.features
    out = []
    for i in range(len(rows)):
        contrib = sorted(((vals[i, k], names[k]) for k in range(len(names)) if names[k] in FEATURE_TEXT),
                         reverse=True)
        picked = []
        for v, n in contrib:
            if v <= 0 or len(picked) == top:
                break
            if n in ("same_city", "same_state", "same_zip3") and rows.iloc[i][n] < 0.5:
                continue
            txt = FEATURE_TEXT[n]
            if txt not in picked:
                picked.append(txt)
        out.append("; ".join(picked) if picked else "a good overall match")
    return out
