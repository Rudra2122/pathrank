"""Load the Kaggle CareerBuilder Job Recommendation Challenge files into a standard schema.

Expected raw files (unzipped) in cfg.data.raw_dir:
  apps.tsv          UserID WindowID Split ApplicationDate JobID
  users.tsv         UserID WindowID Split City State Country ZipCode DegreeType Major
                    GraduationDate WorkHistoryCount TotalYearsExperience CurrentlyEmployed
                    ManagedOthers ManagedHowMany
  user_history.tsv  UserID WindowID Split Sequence JobTitle
  jobs.tsv          JobID WindowID Title Description Requirements City State Country Zip5
                    StartDate EndDate
(The competition ships jobs as jobs.zip -> jobs.tsv; some rows are malformed and are skipped.)

The synthetic generator writes the same files, so this is the only loader.
"""
from __future__ import annotations

import csv
import html
import re
from pathlib import Path

import pandas as pd

from ..utils import log

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _clean_html(s: pd.Series, n: int) -> pd.Series:
    def f(x):
        if not isinstance(x, str):
            return ""
        x = _TAG.sub(" ", html.unescape(html.unescape(x)))
        return _WS.sub(" ", x).strip()[:n]
    return s.map(f)


def _read(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip",
                       usecols=usecols, dtype=str, encoding_errors="replace", low_memory=False)


def _find(raw: Path, name: str) -> Path:
    for cand in [raw / name, raw / name.replace(".tsv", ""), *raw.rglob(name)]:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"{name} not found under {raw}. Run scripts/download_data.sh first.")


def _zip3(s: pd.Series) -> pd.Series:
    return s.fillna("").str.extract(r"(\d{3})", expand=False).fillna("")


def load_raw(raw_dir: str, desc_chars: int = 600, max_users: int | None = None) -> dict[str, pd.DataFrame]:
    raw = Path(raw_dir)

    apps = _read(_find(raw, "apps.tsv"), ["UserID", "ApplicationDate", "JobID"])
    apps = apps.rename(columns={"UserID": "user_id", "JobID": "job_id", "ApplicationDate": "ts"})
    apps["ts"] = pd.to_datetime(apps["ts"], errors="coerce")
    apps["user_id"] = pd.to_numeric(apps["user_id"], errors="coerce")
    apps["job_id"] = pd.to_numeric(apps["job_id"], errors="coerce")
    apps = apps.dropna().astype({"user_id": "int64", "job_id": "int64"})
    apps = apps.sort_values("ts").drop_duplicates(["user_id", "job_id"], keep="first")
    log.info("apps: %s rows", f"{len(apps):,}")

    if max_users:
        top = apps["user_id"].value_counts().head(max_users).index
        apps = apps[apps["user_id"].isin(top)]
        log.info("subsampled to %s most active users -> %s apps", f"{max_users:,}", f"{len(apps):,}")

    u = _read(_find(raw, "users.tsv"))
    users = pd.DataFrame({
        "user_id": pd.to_numeric(u["UserID"], errors="coerce"),
        "city": u["City"].fillna("").str.strip().str.lower(),
        "state": u["State"].fillna("").str.strip().str.upper(),
        "country": u["Country"].fillna("").str.strip().str.upper(),
        "zip3": _zip3(u["ZipCode"]),
        "degree": u["DegreeType"].fillna("None").replace({"": "None"}),
        "major": u["Major"].fillna("").str.strip().str.lower(),
        "work_history_count": pd.to_numeric(u["WorkHistoryCount"], errors="coerce").fillna(0),
        "years_exp": pd.to_numeric(u["TotalYearsExperience"], errors="coerce").fillna(0).clip(0, 60),
        "employed": (u["CurrentlyEmployed"].fillna("").str.lower() == "yes").astype("int8"),
        "managed_others": (u["ManagedOthers"].fillna("").str.lower() == "yes").astype("int8"),
    }).dropna(subset=["user_id"])
    users["user_id"] = users["user_id"].astype("int64")
    users = users.drop_duplicates("user_id")

    try:
        uh = _read(_find(raw, "user_history.tsv"), ["UserID", "Sequence", "JobTitle"])
        uh["user_id"] = pd.to_numeric(uh["UserID"], errors="coerce")
        uh["seq"] = pd.to_numeric(uh["Sequence"], errors="coerce")
        uh = uh.dropna(subset=["user_id", "JobTitle"]).sort_values(["user_id", "seq"])
        uh = uh[uh.groupby("user_id").cumcount() < 10]
        hist = uh.groupby("user_id")["JobTitle"].agg(" | ".join)
        users["work_titles"] = users["user_id"].map(hist).fillna("")
    except FileNotFoundError:
        log.warning("user_history.tsv missing; work_titles empty")
        users["work_titles"] = ""

    j = _read(_find(raw, "jobs.tsv"))
    jobs = pd.DataFrame({
        "job_id": pd.to_numeric(j["JobID"], errors="coerce"),
        "title": j["Title"].fillna("").str.strip(),
        "description": _clean_html(j["Description"], desc_chars),
        "requirements": _clean_html(j["Requirements"], 300),
        "city": j["City"].fillna("").str.strip().str.lower(),
        "state": j["State"].fillna("").str.strip().str.upper(),
        "country": j["Country"].fillna("").str.strip().str.upper(),
        "zip3": _zip3(j["Zip5"]),
        "start_date": pd.to_datetime(j["StartDate"], errors="coerce"),
        "end_date": pd.to_datetime(j["EndDate"], errors="coerce"),
    }).dropna(subset=["job_id"])
    jobs["job_id"] = jobs["job_id"].astype("int64")
    jobs = jobs.drop_duplicates("job_id")
    log.info("jobs: %s rows (after skipping malformed lines)", f"{len(jobs):,}")

    # Keep only interactions whose user and job we can describe.
    before = len(apps)
    apps = apps[apps["user_id"].isin(users["user_id"]) & apps["job_id"].isin(jobs["job_id"])]
    log.info("apps with known user+job: %s (dropped %s)", f"{len(apps):,}", f"{before - len(apps):,}")

    # Missing posting windows: fall back to first/last application seen +/- slack.
    seen = apps.groupby("job_id")["ts"].agg(["min", "max"])
    jobs = jobs.join(seen, on="job_id")
    jobs["start_date"] = jobs["start_date"].fillna(jobs["min"] - pd.Timedelta(days=1))
    jobs["end_date"] = jobs["end_date"].fillna(jobs["max"] + pd.Timedelta(days=14))
    jobs["end_date"] = jobs[["end_date", "max"]].max(axis=1)
    jobs = jobs.drop(columns=["min", "max"]).dropna(subset=["start_date", "end_date"])

    if max_users:
        users = users[users["user_id"].isin(apps["user_id"])]
    return {"apps": apps.reset_index(drop=True), "users": users.reset_index(drop=True),
            "jobs": jobs.reset_index(drop=True)}
