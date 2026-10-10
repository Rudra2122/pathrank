"""XING RecSys Challenge 2016 job interactions -> PathRank's standard schema.

Real job-platform data: 8.8M interactions, 785K users, 1.03M job postings, Aug-Nov 2015.
  interactions.csv   user_id  item_id  interaction_type  created_at   (tab-separated)
  interaction_type   1 click, 2 bookmark, 3 reply (apply), 4 delete (negative)

What this public copy does NOT have: job titles/descriptions, user profiles, locations.
An "application" = a positive interaction (cfg.data.xing_types, default click/bookmark/reply) on a
user-job pair the user never deleted. See data/interactions.py for how missing text is handled.

Source: unofficial mirror of the challenge data (RecoHut-Datasets/xing, branch v1). XING released
it for the RecSys Challenge; use it for research/portfolio only and cite the challenge.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from ..utils import log
from .interactions import to_schema

REPO = "https://github.com/RecoHut-Datasets/xing.git"
URL = "https://github.com/RecoHut-Datasets/xing/raw/v1/xing.zip"
TYPES = {1: "click", 2: "bookmark", 3: "reply", 4: "delete"}


def _find(raw: Path) -> Path | None:
    hits = sorted(raw.rglob("interactions.csv"))
    return hits[0] if hits else None


def download(raw_dir: str) -> Path:
    raw = Path(raw_dir)
    if (p := _find(raw)) is not None:
        return p
    raw.mkdir(parents=True, exist_ok=True)
    z = raw / "xing.zip"
    if not z.exists():
        log.info("downloading XING 2016 interactions (~86 MB) from %s", URL)
        tmp = z.with_suffix(".part")
        try:
            urllib.request.urlretrieve(URL, tmp)
            tmp.rename(z)
        except OSError as e:   # HTTPError is an OSError; some proxies allow git but not raw downloads
            log.warning("direct download failed (%s); trying git clone", e)
            tmp.unlink(missing_ok=True)
            with tempfile.TemporaryDirectory() as td:
                subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", "v1", REPO, td], check=True)
                shutil.move(str(Path(td) / "xing.zip"), z)
    with zipfile.ZipFile(z) as f:
        f.extractall(raw, members=[m for m in f.namelist() if m.endswith("interactions.csv")])
    z.unlink()
    p = _find(raw)
    if p is None:
        raise SystemExit(f"interactions.csv not found after unzipping into {raw}")
    return p


def load_raw(raw_dir: str, types=(1, 2, 3), max_users: int | None = None,
             min_user_apps: int = 2) -> dict[str, pd.DataFrame]:
    p = _find(Path(raw_dir))
    if p is None:
        raise FileNotFoundError(f"interactions.csv not found under {raw_dir}; run the data stage")
    d = pd.read_csv(p, sep="\t", dtype={"user_id": "int64", "item_id": "int64",
                                        "interaction_type": "int8", "created_at": "int64"})
    log.info("xing interactions: %s rows (%s)", f"{len(d):,}",
             ", ".join(f"{TYPES.get(k, k)}={v:,}" for k, v in d.interaction_type.value_counts().sort_index().items()))
    deleted = d.loc[d.interaction_type == 4, ["user_id", "item_id"]].drop_duplicates()
    pos = d[d.interaction_type.isin(list(types))]
    pos = pos.merge(deleted.assign(_del=1), on=["user_id", "item_id"], how="left")
    pos = pos[pos._del.isna()].drop(columns="_del")
    apps = pd.DataFrame({"user_id": pos.user_id.to_numpy(), "job_id": pos.item_id.to_numpy(),
                         "ts": pd.to_datetime(pos.created_at.to_numpy(), unit="s")})
    return to_schema(apps, max_users, min_user_apps)
