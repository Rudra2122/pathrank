"""Static analytics dashboard: weekly cohort retention, apply funnel, notification targeting.

Click-through is not in the dataset (no notification logs exist), so the "notification CTR"
panel shows the targeting curve: share of upcoming applicants reached vs share of users
notified. The API exports real send/click counters to Prometheus once it is live.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def cohort_retention(apps: pd.DataFrame) -> pd.DataFrame:
    a = apps.assign(week=apps.ts.dt.to_period("W").dt.start_time)
    first = a.groupby("uidx")["week"].min().rename("cohort")
    a = a.join(first, on="uidx")
    a["age"] = ((a.week - a.cohort).dt.days // 7).astype(int)
    tab = a.groupby(["cohort", "age"])["uidx"].nunique().unstack(fill_value=0)
    ret = tab.div(tab[0], axis=0).astype(float)
    # weeks that haven't happened yet for late cohorts are unknown, not zero
    last = a.week.max()
    for cohort in ret.index:
        ret.loc[cohort, ret.columns[cohort + pd.to_timedelta(ret.columns * 7, "D") > last]] = np.nan
    return ret


def build(apps: pd.DataFrame, scored_notify: pd.DataFrame | None, results: dict, path: str) -> None:
    ret = cohort_retention(apps)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("#e6e6e6")
    vmax = float(np.nanmax(ret.iloc[:, 1:].to_numpy())) if ret.shape[1] > 1 else 1.0
    im = ax.imshow(np.ma.masked_invalid(ret.iloc[:, :10].to_numpy()), cmap=cmap, vmin=0, vmax=min(1, vmax * 1.2 + 1e-9))
    ax.set_yticks(range(len(ret)), [d.strftime("%b %d") for d in ret.index])
    ax.set_xlabel("weeks since first application")
    ax.set_title("Weekly cohort retention (gray = not yet observed)")
    fig.colorbar(im)
    retention_png = _img(fig)

    daily = apps.set_index("ts").resample("D").agg(apps=("jidx", "size"), users=("uidx", "nunique"))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(daily.index, daily.apps, label="applications")
    ax.plot(daily.index, daily.users, label="active users")
    ax.legend(frameon=False)
    ax.set_title("Daily activity")
    daily_png = _img(fig)

    curve_png = ""
    if scored_notify is not None:
        s = scored_notify.sort_values("p_apply", ascending=False)
        y = s.y_apply.to_numpy()
        x = np.arange(1, len(y) + 1) / len(y)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x, np.cumsum(y) / max(y.sum(), 1), label="model")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="random")
        ax.set_xlabel("share of users notified")
        ax.set_ylabel("share of upcoming applicants reached")
        ax.legend(frameon=False)
        ax.set_title("Notification targeting")
        curve_png = _img(fig)

    rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" if isinstance(v, float) else f"<tr><td>{k}</td><td>{v}</td></tr>"
                   for k, v in results.items() if not isinstance(v, (dict, list)))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PathRank analytics</title>
<style>body{{font-family:system-ui;max-width:900px;margin:24px auto;padding:0 16px;color:#222}}
img{{max-width:100%}}table{{border-collapse:collapse}}td{{border-bottom:1px solid #ddd;padding:4px 10px}}</style></head>
<body><h1>PathRank analytics</h1>
<h2>Retention</h2><img src="data:image/png;base64,{retention_png}">
<h2>Activity</h2><img src="data:image/png;base64,{daily_png}">
{'<h2>Notification targeting</h2><img src="data:image/png;base64,' + curve_png + '">' if curve_png else ''}
<h2>Headline metrics</h2><table>{rows}</table></body></html>"""
    with open(path, "w") as f:
        f.write(html)
