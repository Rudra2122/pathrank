"""OLX Jobs Interactions (Kaggle: olxdatascience/olx-jobs-interactions) -> PathRank schema.

Real job-board data from olx.pl/praca: 65,502,201 events, 3,295,942 users, 185,395 job ads,
two weeks of 2020 (ids randomized, timestamps shifted by a constant).
  interactions.csv   user, item, event, timestamp (unix seconds)
  event              click, bookmark, chat_click, contact_phone_click_1/2/3,
                     contact_partner_click, contact_chat
An "application" here = any configured event (cfg.data.olx_events; default all of them) on a
user-job pair. Interactions only: see data/interactions.py for what that means for the models.
License: non-commercial research use with attribution (see the Kaggle page). Cite:
Kwiecinski et al., "Job Recommendations: Benchmarking of Collaborative Filtering Methods for
Classifieds", Electronics 13, 3049 (2024).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from ..utils import log
from .interactions import to_schema

SLUG = "olxdatascience/olx-jobs-interactions"
EVENTS = ["click", "bookmark", "chat_click", "contact_phone_click_1", "contact_phone_click_2",
          "contact_phone_click_3", "contact_partner_click", "contact_chat"]


def _find(raw: Path) -> Path | None:
    hits = sorted(raw.rglob("interactions.csv"))
    return hits[0] if hits else None


def download(raw_dir: str) -> Path:
    """Kaggle CLI download (needs a token: ~/.kaggle/access_token or KAGGLE_API_TOKEN)."""
    raw = Path(raw_dir)
    if (p := _find(raw)) is not None:
        return p
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI not found: pip install 'kaggle>=1.8', then rerun the data stage")
    raw.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s (~2.1 GB unzipped) with the Kaggle CLI", SLUG)
    subprocess.run(["kaggle", "datasets", "download", "-d", SLUG, "-p", str(raw), "--unzip"], check=True)
    p = _find(raw)
    if p is None:
        raise SystemExit(f"interactions.csv not found under {raw} after download")
    return p


def load_raw(raw_dir: str, events=tuple(EVENTS), max_users: int | None = None,
             seed: int = 0) -> dict[str, pd.DataFrame]:
    p = _find(Path(raw_dir))
    if p is None:
        raise FileNotFoundError(f"interactions.csv not found under {raw_dir}; run the data stage")
    d = pd.read_csv(p, usecols=["user", "item", "event", "timestamp"],
                    dtype={"user": "int64", "item": "int64", "event": "category", "timestamp": "int64"})
    log.info("olx events: %s rows; %s", f"{len(d):,}",
             ", ".join(f"{k}={v:,}" for k, v in d.event.value_counts().items()))
    unknown = set(events) - set(d.event.cat.categories)
    if unknown:
        log.warning("configured events not in the file: %s", sorted(unknown))
    d = d[d.event.isin(list(events))]
    apps = pd.DataFrame({"user_id": d.user.to_numpy(), "job_id": d.item.to_numpy(),
                         "ts": pd.to_datetime(d.timestamp.to_numpy(), unit="s")})
    del d
    return to_schema(apps, max_users, min_user_apps=2, seed=seed)
