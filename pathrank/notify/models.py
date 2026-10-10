"""Notification models.

classifier:  P(user applies within `horizon_days` after snapshot t)
regression:  days until the user's next application, fit on users who return within
             `return_cap_days` (right-censored users are excluded; noted in RESULTS)

Snapshots are taken every `snapshot_every_days`; training snapshots are strictly earlier than
the test snapshot and their label windows end before it starts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score

NOTIFY_FEATURES = ["days_since_last", "n_apps_7d", "n_apps_14d", "n_apps_30d", "n_apps_total",
                   "active_days_30d", "days_since_first", "mean_gap_days", "degree", "years_exp",
                   "employed", "dow"]


def snapshot(store, t: pd.Timestamp, horizon: int, cap: int) -> pd.DataFrame:
    a = store.apps
    h = a[a.ts < t]
    users = h.uidx.unique()
    g = h.groupby("uidx")["ts"]
    df = pd.DataFrame(index=pd.Index(users, name="uidx"))
    last, first = g.max(), g.min()
    day = lambda x: x.dt.total_seconds() / 86400  # noqa: E731
    df["days_since_last"] = day(t - last)
    df["days_since_first"] = day(t - first)
    for w in (7, 14, 30):
        df[f"n_apps_{w}d"] = h[h.ts >= t - pd.Timedelta(days=w)].groupby("uidx").size()
    df["n_apps_total"] = g.size()
    df["active_days_30d"] = h[h.ts >= t - pd.Timedelta(days=30)].assign(d=lambda x: x.ts.dt.floor("D")) \
        .groupby("uidx")["d"].nunique()
    df["mean_gap_days"] = df["days_since_first"] / df["n_apps_total"].clip(lower=1)
    df = df.fillna(0)
    u = store.users.iloc[df.index.to_numpy()]
    df["degree"] = u.degree_ord.to_numpy()
    df["years_exp"] = u.years_exp.to_numpy()
    df["employed"] = u.employed.to_numpy()
    df["dow"] = t.dayofweek
    fut = a[(a.ts >= t) & (a.ts < t + pd.Timedelta(days=max(horizon, cap)))]
    nxt = fut.groupby("uidx")["ts"].min()
    d_next = day(nxt - t).reindex(df.index)
    df["y_apply"] = (d_next < horizon).fillna(False).astype(int)
    df["y_days"] = d_next.where(d_next < cap)
    df["t"] = t
    return df.reset_index()


def build_snapshots(store, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = store.cutoffs
    n = cfg.notify
    gap = pd.Timedelta(days=max(n.horizon_days, n.return_cap_days))
    test_t = c["test"]
    if test_t + gap > c["end"]:
        test_t = (c["end"] - gap).floor("D")
    first = c["start"] + pd.Timedelta(days=n.warmup_days)
    if first > test_t - gap:                      # short timeline (e.g. OLX: two weeks)
        first = c["start"].ceil("D")
    train_ts = pd.date_range(first, test_t - gap, freq=f"{n.snapshot_every_days}D")
    if len(train_ts) == 0:
        raise ValueError(f"no notification training snapshots between {first} and {test_t - gap}: "
                         "lower notify.warmup_days / horizon_days / return_cap_days for short timelines")
    train = pd.concat([snapshot(store, t, n.horizon_days, n.return_cap_days) for t in train_ts], ignore_index=True)
    test = snapshot(store, test_t, n.horizon_days, n.return_cap_days)
    return train, test


def fit_eval(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> tuple[dict, dict]:
    X, Xt = train[NOTIFY_FEATURES], test[NOTIFY_FEATURES]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True,
                                         random_state=seed).fit(X, train.y_apply)
    p = clf.predict_proba(Xt)[:, 1]
    base = -test["days_since_last"].to_numpy()  # recency-only heuristic baseline
    m = {"clf_auc": roc_auc_score(test.y_apply, p), "clf_pr_auc": average_precision_score(test.y_apply, p),
         "clf_base_rate": float(test.y_apply.mean()),
         "recency_baseline_auc": roc_auc_score(test.y_apply, base),
         "recency_baseline_pr_auc": average_precision_score(test.y_apply, base),
         "n_train_rows": int(len(train)), "n_test_users": int(len(test))}
    tr_r, te_r = train.dropna(subset=["y_days"]), test.dropna(subset=["y_days"])
    reg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                        early_stopping=True, random_state=seed).fit(tr_r[NOTIFY_FEATURES], tr_r.y_days)
    d = reg.predict(te_r[NOTIFY_FEATURES])
    med = np.full(len(te_r), tr_r.y_days.median())
    m |= {"reg_mae_days": mean_absolute_error(te_r.y_days, d),
          "reg_rmse_days": float(np.sqrt(mean_squared_error(te_r.y_days, d))),
          "reg_median_baseline_mae": mean_absolute_error(te_r.y_days, med),
          "reg_n_test_returners": int(len(te_r))}
    test = test.assign(p_apply=p)
    test["pred_days"] = reg.predict(Xt)
    return {"clf": clf, "reg": reg, "scored_test": test}, m
