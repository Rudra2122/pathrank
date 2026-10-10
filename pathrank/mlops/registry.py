"""MLflow Model Registry + champion/challenger promotion gate + model card.

Flow per pipeline run:
  1. log the two-tower (PyTorch) and ranker (LightGBM) as MLflow models, plus the Triton
     model repository, feature manifest and model card, tagged with data lineage
     (feature-store content hash, git commit, data source)
  2. register both; the new versions get alias "challenger"
  3. gate: re-score the CURRENT champion ranker on THIS run's held-out set (same users,
     same candidates) and run a paired bootstrap; promote the challenger to "champion"
     only if the CI lower bound of (challenger - champion) NDCG@10 is above the margin.
     First run (no champion) promotes automatically.
Serving resolves models:/<name>@champion, so rollback = moving the alias back.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd

from ..utils import log


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def _names(cfg) -> dict:
    p = f"{cfg.mlops.registry_prefix}-{cfg.data.source}"
    return {"ranker": f"{p}-ranker", "two_tower": f"{p}-two-tower"}


def ranker_eval(ranker_model, eval_df: pd.DataFrame, features: list[str], positives: dict, users) -> pd.DataFrame:
    from ..eval.metrics import per_user_metrics

    df = eval_df[["uidx", "jidx"]].copy()
    df["score"] = ranker_model.predict(eval_df[features])
    return per_user_metrics(df, positives, users)


def register_and_gate(cfg, store, results: dict) -> dict:
    import mlflow
    import mlflow.lightgbm
    from mlflow.tracking import MlflowClient

    from ..eval.ab import replay_ab

    d = cfg.art_dir
    names = _names(cfg)
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(f"pathrank-{cfg.data.source}")
    client = MlflowClient()

    import pickle
    ranker = pickle.load(open(d / "ranker.pkl", "rb"))
    eval_df = pd.read_parquet(d / "eval_set.parquet")
    pos_df = pd.read_parquet(d / "eval_positives.parquet")
    positives = pos_df.groupby("uidx")["jidx"].agg(set).to_dict()
    users = np.sort(eval_df.uidx.unique())

    lineage = {"feature_version": cfg.feature_version, "feature_content_hash": store.manifest["content_hash"],
               "data_source": cfg.data.source, "git_sha": _git_sha(), "text_backend": store.manifest["text_backend"]}
    test = results.get("rank", {}).get("test", {}).get("ranker_v2", {})
    with mlflow.start_run(run_name=f"register-{cfg.feature_version}") as run:
        mlflow.set_tags(lineage | {"stage": "register"})
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in test.items() if isinstance(v, float)})
        reqs = ["lightgbm", "numpy", "pandas"]
        lgb_info = mlflow.lightgbm.log_model(ranker.model, name="ranker", pip_requirements=reqs)
        # the registered retrieval model is the ONNX user tower Triton serves (weights attached too)
        import mlflow.onnx
        import onnx
        onnx_path = d / "model_repository" / "user_tower" / "1" / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError("run the export stage before register")
        pt_info = mlflow.onnx.log_model(onnx.load(str(onnx_path)), name="two_tower",
                                        pip_requirements=["onnxruntime", "numpy"])
        mlflow.log_artifact(str(d / "two_tower.pt"), artifact_path="two_tower_torch")
        mlflow.log_dict({"features": ranker.features, "best_iteration": ranker.model.best_iteration},
                        "ranker/feature_contract.json")
        for p in ("model_repository", "feature_repo"):
            if (d / p).exists():
                mlflow.log_artifacts(str(d / p), artifact_path=p)
        run_id = run.info.run_id

    out = {"run_id": run_id, "lineage": lineage}
    versions = {}
    for key, info in (("ranker", lgb_info), ("two_tower", pt_info)):
        mv = mlflow.register_model(info.model_uri, names[key], tags=lineage)
        client.set_registered_model_alias(names[key], "challenger", mv.version)
        versions[key] = mv.version
    out["versions"] = versions

    # ---- promotion gate (ranker) -----------------------------------------------
    try:
        champ = client.get_model_version_by_alias(names["ranker"], "champion")
    except Exception:  # noqa: BLE001  (no champion yet)
        champ = None
    challenger_pu = ranker_eval(ranker.model, eval_df, ranker.features, positives, users)
    if champ is None:
        decision = {"promoted": True, "reason": "first registered model"}
    else:
        champ_model = mlflow.lightgbm.load_model(f"models:/{names['ranker']}@champion")
        champ_feats = champ_model.feature_name()
        if not set(champ_feats) <= set(eval_df.columns):
            decision = {"promoted": True, "reason": f"champion v{champ.version} needs features missing from this run"}
        else:
            df = eval_df[["uidx", "jidx"]].copy()
            df["score"] = champ_model.predict(eval_df[champ_feats])
            from ..eval.metrics import per_user_metrics
            champ_pu = per_user_metrics(df, positives, users)
            ab = replay_ab(champ_pu, challenger_pu, cfg.mlops.promote_metric, cfg.seed)
            lo = ab["paired_ci95"][0]
            promoted = lo > cfg.mlops.promote_margin
            decision = {"promoted": bool(promoted), "champion_version": champ.version,
                        "paired_diff": ab["paired_diff"], "paired_ci95": ab["paired_ci95"],
                        "reason": ("challenger significantly better" if promoted else
                                   f"CI lower bound {lo:.4f} not above margin {cfg.mlops.promote_margin}")}
    if decision["promoted"]:
        for key in ("ranker", "two_tower"):
            client.set_registered_model_alias(names[key], "champion", versions[key])
    out["gate"] = decision
    out["challenger_ndcg@10"] = float(challenger_pu["ndcg@10"].mean())
    log.info("registry: %s", out)
    return out


def write_model_card(cfg, store, results: dict, reg: dict, path) -> None:
    r = results
    t = r.get("rank", {}).get("test", {})
    fz = r.get("rank", {}).get("fairness", {})
    lines = [
        f"# Model card: PathRank ({cfg.data.source}, feature store {cfg.feature_version})", "",
        f"Registry: ranker v{reg['versions']['ranker']}, two-tower v{reg['versions']['two_tower']}; "
        f"gate: {'PROMOTED' if reg['gate']['promoted'] else 'NOT promoted'} ({reg['gate']['reason']}).",
        f"Lineage: features {reg['lineage']['feature_content_hash']}, git {reg['lineage']['git_sha']}, "
        f"text encoder {reg['lineage']['text_backend']}.", "",
        "## Intended use",
        "Rank open job postings for an early-career job seeker and decide whether to notify them. "
        "Not for screening candidates, setting pay, or any decision about a person's eligibility.", "",
        "## Data",
        f"{store.manifest['n_users']:,} users, {store.manifest['n_jobs']:,} jobs, {store.manifest['n_apps']:,} "
        f"applications; time split {', '.join(f'{k} {v[:10]}' for k, v in store.manifest['cutoffs'].items())}.",
        "Synthetic data: metrics validate the pipeline only." if cfg.data.source == "synthetic" else "", "",
        "## Performance (held-out final window)",
        "| model | ndcg@10 | recall@100 |", "|---|---|---|"]
    for k, v in t.items():
        lines.append(f"| {k} | {v['ndcg@10']:.4f} | {v['recall@100']:.4f} |")
    lines += ["", "## Fairness",
              f"Non-metro jobs: {fz.get('nonmetro_share_of_catalog', 0):.1%} of catalog. Exposure vs NDCG trade-off:",
              "| min non-metro share | ndcg@10 | non-metro exposure |", "|---|---|---|"]
    for s in fz.get("sweep", []):
        lines.append(f"| {s['min_nonmetro_share']} | {s['ndcg@10']:.4f} | {s['nonmetro_exposure']:.1%} |")
    lines += ["", "NDCG@10 by degree type: " + ", ".join(
        f"{k} {v['mean']:.3f}" for k, v in r.get("rank", {}).get("ndcg10_by_degree", {}).items()), "",
        "## Limitations",
        "- Offline evaluation on logged applications: exposure bias of the original site is baked into labels.",
        "- Notification targeting is predictive, not causal (no randomized sends in the data).",
        "- LightGCN only covers users/jobs already in the interaction graph.",
        "- Text embeddings can encode demographic proxies through job titles and majors; monitor by group.", "",
        "## Monitoring",
        "PSI drift on ranker features each run; retraining is triggered when drift exceeds "
        f"{cfg.mlops.drift_psi_retrain} on more than {cfg.mlops.max_drifting_features} features."]
    with open(path, "w") as f:
        f.write("\n".join(x for x in lines if x is not None))
