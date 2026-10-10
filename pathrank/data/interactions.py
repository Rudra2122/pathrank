"""Shared path for interaction-only datasets (XING, OLX): positive user-job pairs -> PathRank schema.

These public datasets have no job text, user profiles or locations. Jobs get a placeholder title
"job <id>"; hashing it makes the two-tower title/history tokens act as job-ID embeddings. Text
vectors are zeros (text.backend=none) and all locations are equal, so the text and location
retrievers are uninformative and the graph + two-tower + popularity signals do the work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import log


def to_schema(apps: pd.DataFrame, max_users: int | None, min_user_apps: int = 2,
              seed: int = 0) -> dict[str, pd.DataFrame]:
    """apps: user_id, job_id, ts (positive interactions, any order, may contain repeats)."""
    apps = apps.sort_values("ts", kind="stable").drop_duplicates(["user_id", "job_id"], keep="first")
    n_u = apps.user_id.value_counts()
    keep = n_u.index[n_u >= min_user_apps]                 # 1-interaction users can't be split in time
    if max_users and len(keep) > max_users:
        # random users (not the most active) so the subsample keeps the real activity distribution
        keep = np.random.default_rng(seed).choice(keep.to_numpy(), max_users, replace=False)
        log.info("subsampled %s random users with >=%d interactions", f"{max_users:,}", min_user_apps)
    apps = apps[apps.user_id.isin(keep)].reset_index(drop=True)

    uid = np.sort(apps.user_id.unique())
    users = pd.DataFrame({"user_id": uid, "city": "", "state": "", "country": "", "zip3": "",
                          "degree": "None", "major": "", "work_history_count": 0.0, "years_exp": 0.0,
                          "employed": 0, "managed_others": 0, "work_titles": ""})
    users = users.astype({"employed": "int8", "managed_others": "int8"})

    seen = apps.groupby("job_id")["ts"].agg(["min", "max"])
    jobs = pd.DataFrame({"job_id": seen.index.to_numpy(),
                         "title": "job " + seen.index.astype(str),
                         "description": "", "requirements": "", "city": "", "state": "", "country": "",
                         "zip3": "",
                         # no posting dates: open from 1 day before the first interaction until 14 days
                         # after the last one (the same fallback the CareerBuilder loader uses)
                         "start_date": (seen["min"] - pd.Timedelta(days=1)).to_numpy(),
                         "end_date": (seen["max"] + pd.Timedelta(days=14)).to_numpy()})
    log.info("%s users, %s jobs, %s positive user-job pairs", f"{len(users):,}", f"{len(jobs):,}", f"{len(apps):,}")
    return {"apps": apps, "users": users, "jobs": jobs}
