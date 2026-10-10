"""Single source of configuration. Loaded from YAML, overridable from the CLI."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    source: str = "careerbuilder"          # careerbuilder | synthetic | xing | olx
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    max_users: int | None = None           # subsample to the N most active users (MacBook mode)
    desc_chars: int = 600                  # truncate descriptions for the text encoder
    synth_users: int = 200_000             # only used when source=synthetic
    synth_jobs: int = 300_000
    synth_apps: int = 2_000_000
    xing_types: list = field(default_factory=lambda: [1, 2, 3])   # click, bookmark, reply
    olx_events: list | None = None         # None = all 8 OLX event types


@dataclass
class SplitCfg:
    # time quantiles over the application timeline
    ctx: float = 0.60    # retriever features as-of here, targets in [ctx, val)
    val: float = 0.80    # ranker trained as-of here, labels in [val, test)
    test: float = 0.90   # final eval as-of here, labels in [test, end]


@dataclass
class TextCfg:
    backend: str = "auto"                  # auto | st | tfidf | none (zero vectors; data without text)
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tfidf_dim: int = 128
    tfidf_fit_rows: int = 200_000


@dataclass
class TwoTowerCfg:
    dim: int = 64
    hidden: int = 256
    epochs: int = 3
    batch_size: int = 1024
    lr: float = 1e-3
    temperature: float = 0.05
    logq_correction: bool = True
    hash_buckets: int = 2 ** 18
    max_train_pairs: int | None = None
    device: str = "auto"                   # auto | cuda | cpu | mps


@dataclass
class LightGCNCfg:
    enabled: bool = True
    dim: int = 64
    layers: int = 3
    epochs: int = 20
    batch_size: int = 16384   # each step propagates the full graph, so fewer, larger steps
    lr: float = 1e-2
    reg: float = 1e-4
    device: str = "auto"                   # auto | cuda | cpu  (sparse ops: no MPS)


@dataclass
class RetrievalCfg:
    k_two_tower: int = 300
    k_text: int = 150
    k_gcn: int = 100
    k_locpop: int = 50


@dataclass
class RankerCfg:
    train_users: int = 10_000
    eval_users: int = 10_000
    top_k_rank: int = 50
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 500
    early_stopping: int = 50


@dataclass
class FairnessCfg:
    metro_share: float = 0.5               # cities covering this share of catalog jobs = "metro"
    min_shares: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4])


@dataclass
class NotifyCfg:
    horizon_days: int = 7                  # classifier: applies within N days
    return_cap_days: int = 14              # regression: days to return, among returners within this
    weekly_cap: int = 2
    snapshot_every_days: int = 3
    warmup_days: int = 14                  # history before the first training snapshot


@dataclass
class AgentCfg:
    provider: str = "auto"                 # auto | anthropic | template
    model: str = "claude-sonnet-5-5"
    eval_users: int = 50
    max_retries: int = 1
    max_concurrency: int = 4               # agent eval cases in flight (LLM latency-bound)
    min_delivered: float = 0.9             # eval gates recorded in results (not pipeline-fatal)
    max_regression: float = 0.05


@dataclass
class FeatureStoreCfg:
    enabled: bool = True
    online: str = "sqlite"                 # sqlite | redis
    redis_url: str = "localhost:6379"
    serve_from_online: bool = True         # API reads user features from the online store
    max_users: int | None = None           # cap users materialized at the serving snapshot
    pit_users: int = 2000                  # users snapshotted at earlier cutoffs (PIT demo/tests)
    chunk: int = 50_000


@dataclass
class TritonCfg:
    url: str = ""                          # e.g. localhost:8000; empty = in-process PyTorch
    max_batch: int = 1024
    opset: int = 17


@dataclass
class MLOpsCfg:
    registry_prefix: str = "pathrank"
    min_rows: dict = field(default_factory=lambda: {"apps": 1000, "users": 500, "jobs": 500})
    promote_metric: str = "ndcg@10"
    promote_margin: float = 0.0            # challenger's CI lower bound must exceed this
    drift_psi_retrain: float = 0.25        # any ranker feature above -> retrain flag
    max_drifting_features: int = 2


@dataclass
class Config:
    seed: int = 42
    feature_version: str = "v1"
    artifacts_dir: str = "artifacts"
    mlflow: bool = True
    mlflow_uri: str = "sqlite:///mlflow.db"
    data: DataCfg = field(default_factory=DataCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    text: TextCfg = field(default_factory=TextCfg)
    two_tower: TwoTowerCfg = field(default_factory=TwoTowerCfg)
    lightgcn: LightGCNCfg = field(default_factory=LightGCNCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    ranker: RankerCfg = field(default_factory=RankerCfg)
    fairness: FairnessCfg = field(default_factory=FairnessCfg)
    notify: NotifyCfg = field(default_factory=NotifyCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)
    featurestore: FeatureStoreCfg = field(default_factory=FeatureStoreCfg)
    triton: TritonCfg = field(default_factory=TritonCfg)
    mlops: MLOpsCfg = field(default_factory=MLOpsCfg)

    # ---- derived paths -------------------------------------------------
    @property
    def store_dir(self) -> Path:
        return Path(self.data.processed_dir) / self.data.source / self.feature_version

    @property
    def art_dir(self) -> Path:
        p = Path(self.artifacts_dir) / self.data.source / self.feature_version
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _merge(dc: Any, upd: dict) -> Any:
    for k, v in upd.items():
        if not hasattr(dc, k):
            raise KeyError(f"unknown config key: {k}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], yaml.safe_load(val))
    return cfg
