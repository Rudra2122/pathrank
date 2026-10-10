"""Feast feature repository for PathRank.

Split of responsibilities:
  * transformation  = pathrank.features.pipeline.AsOf  (the one place feature logic lives)
  * storage/serving = Feast: offline store (Parquet snapshots, point-in-time joins) and
                      online store (SQLite locally, Redis in prod) read by the API.
Parity tests (tests/test_featurestore.py) assert that what Feast serves equals what AsOf
computes, offline and online; that is the training/serving-skew guarantee.

Entities: user (join key uidx), job (join key jidx).
Feature views:
  user_dynamic  activity counts, recency, hashed history tokens, profile text embedding,
                numeric profile vector, all as of the snapshot time
  job_dynamic   popularity (7d / all-time), posting age, days left, as of the snapshot time
Static attributes (titles, location codes, job text embeddings) live in the versioned
Parquet tables and are loaded once at startup.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from ..utils import log

USER_FEATURES = ["tok", "text", "num", "n_apps", "days_since_last"]
JOB_FEATURES = ["pop_7d", "pop_all", "age_days", "days_left"]


def repo_dir(cfg) -> Path:
    p = (cfg.art_dir / "feature_repo").resolve()   # absolute: Feast resolves relative to repo_path
    p.mkdir(parents=True, exist_ok=True)
    return p


def repo_config(cfg):
    from feast.repo_config import RepoConfig

    d = repo_dir(cfg)
    if cfg.featurestore.online == "redis":
        from feast.infra.online_stores.redis import RedisOnlineStoreConfig
        online = RedisOnlineStoreConfig(connection_string=cfg.featurestore.redis_url)
    else:
        from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
        online = SqliteOnlineStoreConfig(path=str(d / "online.db"))
    return RepoConfig(project=f"pathrank_{cfg.data.source}_{cfg.feature_version}", registry=str(d / "registry.db"),
                      provider="local", online_store=online, entity_key_serialization_version=3,
                      repo_path=str(d))


def definitions(cfg, text_dim: int):
    from feast import Entity, FeatureView, Field, FileSource, ValueType
    from feast.types import Array, Float32, Int64

    d = repo_dir(cfg)
    user = Entity(name="user", join_keys=["uidx"], value_type=ValueType.INT64,
                  description="job seeker (row index in the versioned users table)")
    job = Entity(name="job", join_keys=["jidx"], value_type=ValueType.INT64,
                 description="job posting (row index in the versioned jobs table)")
    user_fv = FeatureView(
        name="user_dynamic", entities=[user], ttl=timedelta(days=30), online=True,
        schema=[Field(name="tok", dtype=Array(Int64)), Field(name="text", dtype=Array(Float32)),
                Field(name="num", dtype=Array(Float32)), Field(name="n_apps", dtype=Float32),
                Field(name="days_since_last", dtype=Float32)],
        source=FileSource(name="user_snapshots", path=str(d / "user_snapshots.parquet"),
                          timestamp_field="event_timestamp"),
        tags={"text_dim": str(text_dim), "owner": "pathrank"})
    job_fv = FeatureView(
        name="job_dynamic", entities=[job], ttl=timedelta(days=30), online=True,
        schema=[Field(name=f, dtype=Float32) for f in JOB_FEATURES],
        source=FileSource(name="job_snapshots", path=str(d / "job_snapshots.parquet"),
                          timestamp_field="event_timestamp"),
        tags={"owner": "pathrank"})
    return [user, job, user_fv, job_fv]


def write_yaml(cfg) -> None:
    """feature_store.yaml so the Feast CLI (feast ui / feast serve / feast materialize) works too."""
    d = repo_dir(cfg)
    rc = repo_config(cfg)
    online = (f"  type: redis\n  connection_string: \"{cfg.featurestore.redis_url}\"\n"
              if cfg.featurestore.online == "redis" else f"  type: sqlite\n  path: {d / 'online.db'}\n")
    (d / "feature_store.yaml").write_text(
        f"project: {rc.project}\nregistry: {d / 'registry.db'}\nprovider: local\n"
        f"online_store:\n{online}entity_key_serialization_version: 3\n")
    log.info("feast repo at %s", d)


def get_store(cfg):
    from feast import FeatureStore
    return FeatureStore(config=repo_config(cfg))
