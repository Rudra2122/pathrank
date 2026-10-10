"""Feature store parity, data validation, registry promotion gate, registry-backed serving."""
import os
import pickle

import numpy as np
import pandas as pd

from pathrank.utils import load_json


def test_feature_store_parity(built):
    fs = load_json(built.art_dir / "results.json")["featurestore"]
    assert fs["pit_parity_tokens_equal"] and fs["pit_parity_n_apps_equal"]
    assert fs["pit_parity_text_max_abs_diff"] < 1e-6
    assert fs["online_parity_tokens_equal"] and fs["online_parity_activity_equal"] and fs["online_job_parity_equal"]
    assert fs["online_parity_text_max_abs_diff"] < 1e-6
    assert fs["online_user_fetch_ms_p95"] < 50


def test_online_context_serves_ranker(built):
    from pathrank.featurestore.online import OnlineContext
    from pathrank.features.pipeline import AsOf, load_store
    from pathrank.recommender import Recommender
    s = load_store(built)
    c = s.cutoffs
    rec = Recommender.load(s, built, built.art_dir, backend="torch")
    off = AsOf(s, c["test"], c["end"] + pd.Timedelta(seconds=1), built.two_tower.hash_buckets)
    on = OnlineContext(s, built, c["test"])
    users = off.eval_users(30, 2)
    a = rec.rank(off, users).groupby("uidx").head(10)
    b = rec.rank(on, users).groupby("uidx").head(10)
    assert (a.jidx.to_numpy() == b.jidx.to_numpy()).all()       # online == offline, end to end
    np.testing.assert_allclose(a.score.to_numpy(), b.score.to_numpy(), rtol=1e-5, atol=1e-6)


def test_validation_ran_clean(built):
    v = load_json(built.art_dir / "results.json")["validate"]
    assert v["n_checks"] >= 20 and v["errors"] == []


def test_validation_catches_bad_data(built):
    from pathrank.features.pipeline import load_store
    from pathrank.mlops.validate import validate_store
    s = load_store(built)
    s.apps = pd.concat([s.apps, s.apps.head(5)])                 # duplicate applications
    s.users.loc[0, "years_exp"] = 99                             # out of range
    rep = validate_store(s, built.mlops.min_rows)
    failed = {e["check"] for e in rep["errors"]}
    assert {"integrity.no_duplicate_applications", "sanity.years_exp_range"} <= failed


def test_promotion_gate(built):
    """Identical challenger is rejected; a worse champion gets replaced by the better ranker."""
    import mlflow
    import mlflow.lightgbm
    from mlflow.tracking import MlflowClient
    from pathrank.features.pipeline import load_store
    from pathrank.mlops.registry import register_and_gate
    reg0 = load_json(built.art_dir / "results.json")["register"]
    assert reg0["gate"]["promoted"]                              # first model auto-promoted
    s = load_store(built)
    res = load_json(built.art_dir / "results.json")
    same = register_and_gate(built, s, res)
    assert not same["gate"]["promoted"] and same["gate"]["paired_diff"] == 0.0

    # make the weaker ranker v1 the champion, then re-register v2: the gate should promote it
    mlflow.set_tracking_uri(built.mlflow_uri)
    name = f"{built.mlops.registry_prefix}-{built.data.source}-ranker"
    v1 = pickle.load(open(built.art_dir / "ranker_v1.pkl", "rb"))
    with mlflow.start_run(run_name="test-weak-champion"):
        info = mlflow.lightgbm.log_model(v1.model, name="ranker", pip_requirements=["lightgbm"])
    mv = mlflow.register_model(info.model_uri, name)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)
    better = register_and_gate(built, s, res)
    assert better["gate"]["promoted"], better["gate"]
    assert better["gate"]["paired_ci95"][0] > 0


def test_registry_backed_serving(built, monkeypatch):
    from pathrank.serve.service import Service
    monkeypatch.setenv("MODEL_SOURCE", "registry")
    svc = Service.from_artifacts(built)
    assert svc.feature_source == "feast_online"
    uid = int(svc.store.users.user_id.iloc[svc.eval_users(1, 5)[0]])
    recs = svc.recommend(uid, 5)["recommendations"]
    monkeypatch.delenv("MODEL_SOURCE")
    local = Service.from_artifacts(built).recommend(uid, 5)["recommendations"]
    assert [r["job_id"] for r in recs] == [r["job_id"] for r in local]


def test_model_card_and_retrain_flag(built):
    card = (built.art_dir / "MODEL_CARD.md").read_text()
    assert "Intended use" in card and "Fairness" in card and "Lineage" in card
    flag = load_json(built.art_dir / "retrain_flag.json")
    assert set(flag) >= {"retrain_required", "drifting", "threshold_psi"}


def test_api_features_endpoint(built):
    os.environ["PATHRANK_CONFIG"] = "configs/ci.yaml"
    from fastapi.testclient import TestClient
    from pathrank.serve.app import _state, app
    _state.clear()
    c = TestClient(app)
    from pathrank.serve.service import Service
    s = Service.from_artifacts(built)
    uid = int(s.store.users.user_id.iloc[s.eval_users(1, 9)[0]])
    r = c.get(f"/features/{uid}").json()
    assert r["source"] == "feast_online" and r["profile_embedding_dim"] > 0
    h = c.get("/health").json()
    assert h["feature_source"] == "feast_online" and h["towers"] == "torch"


def test_online_store_is_safe_under_concurrent_requests(built):
    """Regression: Feast's SQLite online store shares one connection; concurrent API/agent threads
    used to fail with sqlite3.InterfaceError or report present users as missing."""
    from concurrent.futures import ThreadPoolExecutor

    from pathrank.featurestore.online import OnlineContext
    from pathrank.features.pipeline import load_store
    s = load_store(built)
    on = OnlineContext(s, built, s.cutoffs["test"])
    users = np.where(np.isin(np.arange(len(s.users)), np.arange(len(s.users))))[0][:400]

    def fetch(u):
        return on.user_features(np.array([u]))["tok"].shape

    with ThreadPoolExecutor(8) as pool:
        shapes = list(pool.map(fetch, np.tile(users, 3)))
    assert len(shapes) == 1200


def test_api_requests_are_traced(built):
    """Every request is an `http` span named by route template; recommender stages nest under it."""
    os.environ["PATHRANK_CONFIG"] = "configs/ci.yaml"
    from evalwright.testing import capture_spans
    from fastapi.testclient import TestClient

    from pathrank.serve.app import _state, app
    _state.clear()
    c = TestClient(app)
    from pathrank.serve.service import Service
    uid = int(Service.from_artifacts(built).store.users.user_id.iloc[0])
    c.get("/health")                                           # warm the service outside the capture
    c.get(f"/recommend/{uid}?k=3")
    with capture_spans("api") as cap:
        assert c.get(f"/recommend/{uid}?k=3").status_code == 200
        assert c.get("/recommend/-1").status_code == 404
    http = [s for s in cap.spans if s.type == "http"]
    assert {s.name for s in http} == {"GET /recommend/{user_id}"}
    ok = next(s for s in http if s.metrics["status_code"] == 200)
    tree = cap.tree()
    assert "GET /recommend/{user_id} [http]\n  recommend [task]\n    features.user [retrieval]\n" \
           "    retrieve.candidates [retrieval]\n    features.pairs [function]\n    rank.lightgbm [function]" in tree
    rec = next(s for s in cap.spans if s.name == "recommend" and s.parent_id == ok.span_id)
    assert rec.input["user_id"] == uid and len(rec.output["recommendations"]) == 3
    assert any(s.metrics.get("status_code") == 404 for s in http)


def test_agent_eval_ran_through_evalwright(built):
    a = load_json(built.art_dir / "results.json")["agent"]["evalwright"]
    assert a["experiment_id"] and a["n_spans"] > 0 and a["error_rate"] == 0
    assert set(a["scores"]) == {"delivered", "first_try", "judge_quality"}
    assert a["gate_failures"] == []
