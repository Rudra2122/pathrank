"""Data validation gate. Runs after the feature store build; any ERROR stops the pipeline.

Checks are expectations about the data contract, not tuning knobs:
  volume       minimum rows per table
  integrity    every application points at a known user and job; no duplicate (user, job)
  sanity       ranges (experience, posting windows), finite embeddings
  temporal     cutoffs strictly ordered, every split non-empty, apps inside the timeline
  drift-prone  null/empty rates on the fields models depend on (WARN above thresholds)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_store(store, min_rows: dict) -> dict:
    checks: list[dict] = []

    def add(name, ok, level="ERROR", detail=""):
        checks.append({"check": name, "status": "PASS" if ok else level, "detail": str(detail)})

    u, j, a = store.users, store.jobs, store.apps
    for t, df in (("apps", a), ("users", u), ("jobs", j)):
        add(f"volume.{t}", len(df) >= min_rows.get(t, 1), detail=f"{len(df):,} rows (min {min_rows.get(t, 1):,})")

    add("integrity.apps_user_fk", a.uidx.between(0, len(u) - 1).all(), detail="uidx within users table")
    add("integrity.apps_job_fk", a.jidx.between(0, len(j) - 1).all(), detail="jidx within jobs table")
    dup = int(a.duplicated(["uidx", "jidx"]).sum())
    add("integrity.no_duplicate_applications", dup == 0, detail=f"{dup} duplicates")
    add("integrity.unique_user_ids", u.user_id.is_unique)
    add("integrity.unique_job_ids", j.job_id.is_unique)

    add("sanity.years_exp_range", u.years_exp.between(0, 60).all(), detail=f"max {u.years_exp.max()}")
    bad_win = int((j.end_date < j.start_date).sum())
    add("sanity.posting_window_order", bad_win == 0, detail=f"{bad_win} jobs end before they start")
    add("sanity.job_embeddings_finite", bool(np.isfinite(store.job_vec).all()))
    norms = np.linalg.norm(store.job_vec, axis=1)
    add("sanity.job_embeddings_nonzero", float((norms < 1e-6).mean()) < 0.01, "WARN",
        f"{(norms < 1e-6).mean():.2%} zero vectors")

    c = store.cutoffs
    order = [c["start"], c["ctx"], c["val"], c["test"], c["end"]]
    add("temporal.cutoffs_ordered", all(x < y for x, y in zip(order, order[1:])), detail=[str(x) for x in order])
    for k, v in store.manifest["apps_per_split"].items():
        add(f"temporal.split_nonempty.{k}", v > 0, detail=f"{v:,}")
    add("temporal.apps_within_timeline", bool((a.ts >= c["start"]).all() and (a.ts <= c["end"]).all()))
    # applications should mostly land on jobs open at that moment (posting-window consistency)
    s = a.sample(min(len(a), 200_000), random_state=0)
    jj = j.iloc[s.jidx.to_numpy()]
    inside = ((s.ts.to_numpy() >= jj.start_date.to_numpy() - np.timedelta64(1, "D"))
              & (s.ts.to_numpy() <= jj.end_date.to_numpy() + np.timedelta64(1, "D"))).mean()
    add("temporal.apps_inside_posting_window", inside > 0.9, "WARN", f"{inside:.1%}")

    for col, lim in (("city", 0.2), ("state", 0.1)):
        r = float((u[col].fillna("") == "").mean())
        add(f"completeness.users.{col}", r <= lim, "WARN", f"{r:.1%} empty (limit {lim:.0%})")
    r = float((j.title.fillna("") == "").mean())
    add("completeness.jobs.title", r <= 0.01, "ERROR", f"{r:.2%} empty")
    r = float((j.description.fillna("") == "").mean())
    add("completeness.jobs.description", r <= 0.2, "WARN", f"{r:.1%} empty")

    df = pd.DataFrame(checks)
    return {"n_checks": len(df), "errors": df[df.status == "ERROR"].to_dict("records"),
            "warnings": df[df.status == "WARN"].to_dict("records"), "table": df}


def validate_ranker_frame(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Serving-time contract on ranker inputs: columns present, no +-inf."""
    missing = [f for f in features if f not in df]
    if missing:
        return [f"missing feature columns: {missing}"]
    X = df[features].to_numpy(np.float64)
    return ["infinite values in ranker features"] if np.isinf(X).any() else []
