"""Thin MLflow wrapper: every stage logs params/metrics/artifacts. No-ops when disabled."""
from __future__ import annotations

import os
from contextlib import contextmanager

from .utils import log

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


@contextmanager
def run(cfg, stage: str):
    if not cfg.mlflow:
        yield _Null()
        return
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed; tracking disabled")
        yield _Null()
        return
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(f"pathrank-{cfg.data.source}")
    with mlflow.start_run(run_name=f"{stage}-{cfg.feature_version}"):
        mlflow.set_tag("stage", stage)
        params = {k: str(v)[:250] for k, v in _flatten(cfg.to_dict()).items()}
        mlflow.log_params(params)
        yield _Ml(mlflow)


class _Null:
    def metrics(self, m: dict, step: int | None = None):
        pass

    def artifact(self, path):
        pass


class _Ml:
    def __init__(self, mlflow):
        self.m = mlflow

    def metrics(self, m: dict, step: int | None = None):
        clean = {k.replace("@", "_at_"): float(v) for k, v in _flatten(m).items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v}
        self.m.log_metrics(clean, step=step)

    def artifact(self, path):
        if os.path.exists(path):
            self.m.log_artifact(str(path))
