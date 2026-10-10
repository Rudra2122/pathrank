"""Feature/score drift between a reference window and a current window.

PSI per feature (built in, no dependencies) + an Evidently HTML report when Evidently is
installed (its API changes often across versions, so it is optional and isolated here).
PSI rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 investigate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import log


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [ref.min() - 1e-9, ref.max() + 1e-9]]))
    r = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref)
    c = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(((c - r) * np.log(c / r)).sum())


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        v = psi(ref[col].to_numpy(float), cur[col].to_numpy(float))
        status = "stable" if v < 0.1 else ("watch" if v < 0.25 else "drift")
        rows.append({"feature": col, "psi": round(v, 4), "status": status,
                     "ref_mean": ref[col].mean(), "cur_mean": cur[col].mean()})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, path: str) -> bool:
    try:
        from evidently import Report  # >= 0.7
        from evidently.presets import DataDriftPreset
        snap = Report([DataDriftPreset()]).run(reference_data=ref, current_data=cur)
        snap.save_html(path)
        return True
    except Exception as e:  # noqa: BLE001
        try:
            from evidently.metric_preset import DataDriftPreset  # older API
            from evidently.report import Report
            r = Report(metrics=[DataDriftPreset()])
            r.run(reference_data=ref, current_data=cur)
            r.save_html(path)
            return True
        except Exception:  # noqa: BLE001
            log.info("evidently report skipped (%s); PSI table still written", type(e).__name__)
            return False
