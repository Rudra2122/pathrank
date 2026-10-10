"""PathRank end-to-end pipeline.

  python scripts/run_pipeline.py --config configs/careerbuilder.yaml --stages all
  python scripts/run_pipeline.py --config configs/synthetic_small.yaml --stages data,features,retrieval,rank

Stages (each reads what earlier stages wrote, so they can be re-run independently):
  data       synthetic: generate TSVs   careerbuilder: check the Kaggle files exist
  features   versioned feature store (parquet + text embeddings + manifest)
  retrieval  train two-tower + LightGCN; evaluate each retriever and the baselines
  rank       LightGBM LambdaRank v1/v2, SHAP, replayed A/B, fairness re-rank sweep
  notify     apply classifier + return-time regression + targeting curve
  agent      LangGraph notification agent eval (guardrails + judge + adversarial check)
  monitor    PSI drift ranker-train vs test, Evidently report if installed
  report     RESULTS.md + analytics dashboard
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathrank import tracking  # noqa: E402
from pathrank.config import load_config  # noqa: E402
from pathrank.features.pipeline import AsOf, build_store, load_store  # noqa: E402
from pathrank.utils import load_json, log, save_json, seed_everything, timer  # noqa: E402

ALL = ["data", "features", "validate", "featurestore", "retrieval", "rank", "export", "notify", "agent",
       "monitor", "register", "report"]


def _results_path(cfg) -> Path:
    return cfg.art_dir / "results.json"


def _update_results(cfg, key: str, value: dict) -> None:
    p = _results_path(cfg)
    res = load_json(p) if p.exists() else {}
    res[key] = value
    save_json(res, p)


def _contexts(store, cfg):
    c = store.cutoffs
    hb = cfg.two_tower.hash_buckets
    return {"retr": AsOf(store, c["ctx"], c["val"], hb),
            "rank": AsOf(store, c["val"], c["test"], hb),
            "test": AsOf(store, c["test"], c["end"] + pd.Timedelta(seconds=1), hb)}


# --------------------------------------------------------------------------------------
def stage_data(cfg):
    raw = Path(cfg.data.raw_dir)
    if cfg.data.source == "synthetic":
        from pathrank.data.synthetic import generate
        out = raw / "synthetic"
        if (out / "apps.tsv").exists():
            log.info("synthetic raw data exists at %s; reusing", out)
            return
        with timer("generate_synthetic"):
            stats = generate(str(out), cfg.data.synth_users, cfg.data.synth_jobs, cfg.data.synth_apps, cfg.seed)
        _update_results(cfg, "data", stats)
    elif cfg.data.source in ("xing", "olx"):
        import importlib
        download = importlib.import_module(f"pathrank.data.{cfg.data.source}").download
        with timer(f"download_{cfg.data.source}"):
            p = download(str(raw / cfg.data.source))
        log.info("%s interactions at %s", cfg.data.source, p)
    else:
        need = ["apps.tsv", "users.tsv", "jobs.tsv"]
        missing = [n for n in need if not list((raw / "careerbuilder").rglob(n))]
        if missing:
            raise SystemExit(f"Missing {missing} in {raw/'careerbuilder'}. Run: bash scripts/download_data.sh")
        log.info("careerbuilder raw files present")


def stage_features(cfg):
    with timer("build_feature_store"):
        build_store(cfg)


def stage_validate(cfg):
    from pathrank.mlops.validate import validate_store

    store = load_store(cfg)
    rep = validate_store(store, cfg.mlops.min_rows)
    rep["table"].to_csv(cfg.art_dir / "data_validation.csv", index=False)
    out = {"n_checks": rep["n_checks"], "errors": rep["errors"], "warnings": rep["warnings"]}
    _update_results(cfg, "validate", out)
    log.info("validate: %d checks, %d errors, %d warnings", rep["n_checks"], len(rep["errors"]), len(rep["warnings"]))
    for w in rep["warnings"]:
        log.warning("data WARN %s: %s", w["check"], w["detail"])
    if rep["errors"]:
        raise SystemExit(f"data validation FAILED: {rep['errors']}")


def stage_register(cfg):
    from pathrank.mlops.registry import register_and_gate, write_model_card

    store = load_store(cfg)
    res = load_json(_results_path(cfg))
    reg = register_and_gate(cfg, store, res)
    write_model_card(cfg, store, res, reg, cfg.art_dir / "MODEL_CARD.md")
    import mlflow
    with mlflow.start_run(run_id=reg["run_id"]):
        mlflow.log_artifact(str(cfg.art_dir / "MODEL_CARD.md"))
    _update_results(cfg, "register", reg)


def stage_featurestore(cfg):
    """Feast: write snapshots, apply, materialize, then prove offline + online parity with AsOf."""
    if not cfg.featurestore.enabled:
        log.info("feature store disabled")
        return
    import time
    from pathrank.featurestore.materialize import NAN_SENTINEL, build_feature_store, historical_user_features
    from pathrank.featurestore.online import OnlineContext

    store = load_store(cfg)
    c = store.cutoffs
    hb = cfg.two_tower.hash_buckets
    with timer("feast_materialize"):
        info = build_feature_store(store, cfg)

    # 1) offline point-in-time parity at the val cutoff
    val = AsOf(store, c["val"], c["test"], hb)
    rng = np.random.default_rng(cfg.seed)
    snap = pd.read_parquet(cfg.art_dir / "feature_repo" / "user_snapshots.parquet", columns=["uidx", "event_timestamp"])
    at_val = snap[snap.event_timestamp == pd.Timestamp(c["val"]).tz_localize("UTC")].uidx.to_numpy()
    sample = np.sort(rng.choice(at_val, min(300, len(at_val)), replace=False))
    hist = historical_user_features(cfg, sample, c["val"])
    ref = val.user_features(sample)
    info["pit_rows_returned"] = int(len(hist))
    info["pit_parity_text_max_abs_diff"] = float(np.abs(np.stack(hist.text.to_numpy()) - ref["text"]).max())
    info["pit_parity_tokens_equal"] = bool((np.stack(hist.tok.to_numpy()) == ref["tok"]).all())
    info["pit_parity_n_apps_equal"] = bool((hist.n_apps.to_numpy() == val.user_n_apps[sample]).all())

    # 2) online parity at the serving cutoff, plus fetch latency
    test = AsOf(store, c["test"], c["end"] + pd.Timedelta(seconds=1), hb)
    oc = OnlineContext(store, cfg, c["test"])
    users = np.sort(rng.choice(np.where(test.user_n_apps > 0)[0], 200, replace=False))
    on = oc.user_features(users)
    off = test.user_features(users)
    info["online_parity_text_max_abs_diff"] = float(np.abs(on["text"] - off["text"]).max())
    info["online_parity_tokens_equal"] = bool((on["tok"] == off["tok"]).all())
    info["online_parity_num_max_abs_diff"] = float(np.abs(on["num"] - off["num"]).max())
    last_off = np.where(np.isnan(test.user_last[users]), NAN_SENTINEL, test.user_last[users])
    last_on = np.where(np.isnan(oc.user_last[users]), NAN_SENTINEL, oc.user_last[users])
    info["online_parity_activity_equal"] = bool(np.allclose(last_on, last_off) and
                                                (oc.user_n_apps[users] == test.user_n_apps[users]).all())
    cat = test.catalog
    info["online_job_parity_equal"] = bool(np.allclose(oc.job_pop_7d[cat], test.job_pop_7d[cat]) and
                                           np.allclose(oc.job_pop_all[cat], test.job_pop_all[cat]) and
                                           np.allclose(oc.job_left[cat], test.job_left[cat], atol=1e-3, equal_nan=True))
    lat = []
    for u in users[:100]:
        t0 = time.perf_counter()
        oc.user_features(np.array([u]))
        lat.append((time.perf_counter() - t0) * 1000)
    info["online_user_fetch_ms_p50"] = round(float(np.percentile(lat[1:], 50)), 2)
    info["online_user_fetch_ms_p95"] = round(float(np.percentile(lat[1:], 95)), 2)
    _update_results(cfg, "featurestore", info)
    log.info("featurestore: %s", info)
    bad = [k for k, v in info.items() if (k.endswith("_equal") and v is False) or
           (k.endswith("max_abs_diff") and v > 1e-5)]
    if bad:
        raise SystemExit(f"feature store parity FAILED: {bad}")


def stage_retrieval(cfg):
    import torch
    from pathrank.eval.metrics import per_user_metrics, summarize
    from pathrank.recommender import Recommender
    from pathrank.retrieval.lightgcn import train_lightgcn
    from pathrank.retrieval.simple import popularity
    from pathrank.retrieval.two_tower import train_two_tower

    store = load_store(cfg)
    ctx = _contexts(store, cfg)
    times: dict = {}
    with tracking.run(cfg, "retrieval") as tr:
        with timer("train_two_tower", times):
            tt = train_two_tower(ctx["retr"], cfg.two_tower, cfg.seed, tr)
        gcn = None
        if cfg.lightgcn.enabled:
            edges = store.apps[store.apps.ts < store.cutoffs["val"]]
            with timer("train_lightgcn", times):
                gcn = train_lightgcn(edges, cfg.lightgcn, cfg.seed, tr)
        d = cfg.art_dir
        torch.save(tt.state_dict(), d / "two_tower.pt")
        with open(d / "gcn.pkl", "wb") as f:
            pickle.dump(gcn, f)

        # evaluate each retriever alone on the test window
        T = ctx["test"]
        users = T.eval_users(cfg.ranker.eval_users, cfg.seed)
        pos = T.label_sets(users)
        cat_mask = T.catalog_mask
        pos = {u: {j for j in s if cat_mask[j]} for u, s in pos.items()}
        rec = Recommender(store, cfg, tt, gcn)
        with timer("test_candidates", times):
            uf = T.user_features(users)
            cand, src = rec.candidates(T, users, uf)
        out = {}
        for name, (S, J) in {**src, "popularity": popularity(T, len(users), 100)}.items():
            df = pd.DataFrame({"uidx": np.repeat(users, J.shape[1]), "jidx": J.ravel(), "score": S.ravel()})
            df = df[df.jidx >= 0]
            out[name] = summarize(per_user_metrics(df, pos, users))
        merged = cand.assign(score=-cand[["tt_rank", "txt_rank", "gcn_rank", "lp_rank"]].min(axis=1))
        out["union_candidates"] = summarize(per_user_metrics(merged, pos, users))
        # what the ranker can reach: share of each user's positives anywhere in the candidate set
        got = cand.groupby("uidx")["jidx"].agg(set).to_dict()
        out["union_candidates_recall_all"] = float(np.mean(
            [len(pos[u] & got.get(u, set())) / len(pos[u]) for u in users if pos.get(u)]))
        out["avg_candidates_per_user"] = float(len(cand) / len(users))
        n_test = len(T.labels)
        out["test_apps_in_open_catalog_share"] = float(cat_mask[T.labels.jidx.to_numpy()].mean()) if n_test else 0.0
        out["test_apps_to_jobs_unseen_before_cutoff"] = float(
            (T.job_pop_all[T.labels.jidx.to_numpy()] == 0).mean()) if n_test else 0.0
        if gcn is not None:
            out["gcn_catalog_coverage"] = float(np.isin(T.catalog, gcn.item_ids).mean())
            out["gcn_user_coverage"] = float(np.isin(users, gcn.user_pos.index).mean())
        out["catalog_size_test"] = int(len(T.catalog))
        out["two_tower_device"] = getattr(tt, "train_device", "cpu")
        out["two_tower_pairs_per_s"] = round(getattr(tt, "pairs_per_s", 0.0))
        if gcn is not None:
            out["lightgcn_device"] = getattr(gcn, "train_device", "cpu")
            out["lightgcn_train_s"] = round(getattr(gcn, "train_seconds", 0.0), 1)
        out |= times
        tr.metrics({f"retr.{k}": v for k, v in out.items() if isinstance(v, (int, float))})
        for k, v in out.items():
            if isinstance(v, dict):
                tr.metrics({f"retr.{k}.{m}": x for m, x in v.items()})
        _update_results(cfg, "retrieval", out)
        log.info("retrieval results: %s", {k: (v.get("recall@100") if isinstance(v, dict) else v) for k, v in out.items()})


def stage_rank(cfg):
    import torch
    from pathrank.eval.ab import replay_ab
    from pathrank.eval.metrics import exposure, per_user_metrics, summarize
    from pathrank.ranking.explain import global_importance, reasons
    from pathrank.ranking.features import RANKER_V1, RANKER_V2, attach_labels, pair_features
    from pathrank.ranking.ranker import LGBRanker
    from pathrank.recommender import Recommender
    from pathrank.rerank.fairness import metro_flags, rerank
    from pathrank.retrieval.two_tower import TwoTower

    store = load_store(cfg)
    ctx = _contexts(store, cfg)
    d = cfg.art_dir
    tt = TwoTower(store.job_vec.shape[1], cfg.two_tower)
    tt.load_state_dict(torch.load(d / "two_tower.pt", weights_only=True))
    tt.eval()
    gcn = pickle.load(open(d / "gcn.pkl", "rb"))
    rec = Recommender(store, cfg, tt, gcn)
    res: dict = {}

    with tracking.run(cfg, "rank") as tr:
        # ---- training data: candidates as-of val, labels in [val, test) ----
        R = ctx["rank"]
        tr_users = R.eval_users(cfg.ranker.train_users, cfg.seed + 1)
        with timer("ranker_train_features", res):
            uf = R.user_features(tr_users)
            cand, _ = rec.candidates(R, tr_users, uf)
            train_df = attach_labels(R, pair_features(R, cand, uf))
        res["ranker_train_pos_in_candidates"] = int(train_df.label.sum())
        v1 = LGBRanker(RANKER_V1, cfg.ranker, cfg.seed, "ranker_v1")
        v2 = LGBRanker(RANKER_V2, cfg.ranker, cfg.seed, "ranker_v2")
        with timer("train_rankers", res):
            res["ranker_v1_fit"] = v1.fit(train_df)
            res["ranker_v2_fit"] = v2.fit(train_df)

        # ---- test ----
        T = ctx["test"]
        users = T.eval_users(cfg.ranker.eval_users, cfg.seed)
        pos = {u: {j for j in s if T.catalog_mask[j]} for u, s in T.label_sets(users).items()}
        with timer("test_features", res):
            uft = T.user_features(users)
            tc, src = rec.candidates(T, users, uft)
            test_df = pair_features(T, tc, uft)
        test_df["score_v1"] = v1.predict(test_df)
        test_df["score_v2"] = v2.predict(test_df)

        from pathrank.retrieval.simple import popularity
        S, J = popularity(T, len(users), 100)
        pop_df = pd.DataFrame({"uidx": np.repeat(users, J.shape[1]), "jidx": J.ravel(), "score": S.ravel()})
        lp_df = test_df.assign(score=-test_df.lp_rank).dropna(subset=["lp_rank"])
        tt_df = test_df.assign(score=test_df.tt_score).dropna(subset=["tt_score"])
        pu = {"popularity": per_user_metrics(pop_df, pos, users),
              "location_popularity": per_user_metrics(lp_df, pos, users),
              "two_tower_only": per_user_metrics(tt_df, pos, users),
              "ranker_v1": per_user_metrics(test_df, pos, users, "score_v1"),
              "ranker_v2": per_user_metrics(test_df, pos, users, "score_v2")}
        res["test"] = {k: summarize(v) for k, v in pu.items()}
        base = res["test"]["popularity"]["ndcg@10"]
        lbase = res["test"]["location_popularity"]["ndcg@10"]
        v2n = res["test"]["ranker_v2"]["ndcg@10"]
        res["ndcg10_lift_vs_popularity"] = (v2n - base) / base if base > 0 else None
        res["ndcg10_lift_vs_location_popularity"] = (v2n - lbase) / lbase if lbase > 0 else None

        # cold vs warm users
        cold = set(users[T.user_n_apps[users] == 0])
        pv2 = pu["ranker_v2"]
        res["test_cold_users"] = summarize(pv2[pv2.uidx.isin(cold)]) if cold else {}
        res["test_warm_users"] = summarize(pv2[~pv2.uidx.isin(cold)])

        # user-side fairness: quality by degree type
        deg = store.users.degree.to_numpy()
        res["ndcg10_by_degree"] = pv2.assign(degree=deg[pv2.uidx.to_numpy()]).groupby("degree")["ndcg@10"] \
            .agg(["mean", "size"]).round(4).to_dict("index")

        # ---- replayed A/B v1 vs v2 ----
        res["ab_v1_vs_v2"] = replay_ab(pu["ranker_v1"], pu["ranker_v2"], "ndcg@10", cfg.seed)

        # ---- SHAP ----
        with timer("shap", res):
            imp = global_importance(v2, test_df)
        res["shap_mean_abs"] = imp.round(5).to_dict()
        ranked = test_df.sort_values(["uidx", "score_v2"], ascending=[True, False], kind="stable")
        sample_u = users[:3]
        ex = ranked[ranked.uidx.isin(sample_u)].groupby("uidx").head(3).copy()
        ex["reason"] = reasons(v2, ex)
        ex["title"] = store.jobs.title.to_numpy()[ex.jidx.to_numpy()]
        res["why_this_job_examples"] = ex[["uidx", "title", "score_v2", "reason"]].to_dict("records")

        # ---- fairness re-rank sweep (item side: non-metro job exposure) ----
        prot = ~metro_flags(store, T.catalog, cfg.fairness.metro_share)
        top = ranked.groupby("uidx").head(cfg.ranker.top_k_rank)
        sweep = []
        with timer("fairness_sweep", res):
            for ms in cfg.fairness.min_shares:
                rr = rerank(top, prot, ms, k=10, pool=cfg.ranker.top_k_rank)
                rr["nonmetro"] = prot[rr.jidx.to_numpy()]
                m = summarize(per_user_metrics(rr.assign(score=-rr.pos), pos, users))
                e = exposure(rr, "nonmetro").get(True, 0.0)
                sweep.append({"min_nonmetro_share": ms, "ndcg@10": m["ndcg@10"], "recall@10": m["recall@10"],
                              "nonmetro_exposure": float(e)})
        catalog_share = float(prot[T.catalog].mean())
        label_share = float(np.mean([prot[j] for s in pos.values() for j in s])) if pos else 0.0
        res["fairness"] = {"nonmetro_share_of_catalog": catalog_share,
                           "nonmetro_share_of_test_applications": label_share, "sweep": sweep}

        tr.metrics({f"rank.{k}.{m}": v for k, d_ in res["test"].items() for m, v in d_.items()})
        rec.ranker = v2
        rec.save(d)
        with open(d / "ranker_v1.pkl", "wb") as f:
            pickle.dump(v1, f)
        # reference sample for drift monitoring
        train_df[RANKER_V2].sample(min(200_000, len(train_df)), random_state=cfg.seed).to_parquet(d / "ref_features.parquet")
        test_df[RANKER_V2 + ["score_v2"]].sample(min(200_000, len(test_df)), random_state=cfg.seed).to_parquet(d / "cur_features.parquet")
        # held-out set the registry gate re-scores champion vs challenger on
        test_df[["uidx", "jidx"] + RANKER_V2].to_parquet(d / "eval_set.parquet", index=False)
        pd.DataFrame([(u, j) for u, s_ in pos.items() for j in s_], columns=["uidx", "jidx"]) \
            .to_parquet(d / "eval_positives.parquet", index=False)
    _update_results(cfg, "rank", res)
    log.info("rank: popularity ndcg@10=%.4f  locpop=%.4f  v1=%.4f  v2=%.4f", base, lbase,
             res["test"]["ranker_v1"]["ndcg@10"], v2n)


def stage_export(cfg):
    """ONNX export + Triton model repository + parity + CPU latency (PyTorch vs ONNX Runtime)."""
    import time

    from pathrank.inference.backends import OnnxTowers
    from pathrank.inference.export import build_model_repository
    from pathrank.recommender import Recommender

    store = load_store(cfg)
    c = store.cutoffs
    T = AsOf(store, c["test"], c["end"] + pd.Timedelta(seconds=1), cfg.two_tower.hash_buckets)
    rec = Recommender.load(store, cfg, cfg.art_dir, backend="torch")
    users = T.eval_users(2000, cfg.seed)
    uf = T.user_features(users)
    jf = T.job_features(T.catalog[:4096])
    repo = cfg.art_dir / "model_repository"
    rep = build_model_repository(rec, store, uf, jf, repo, cfg.triton.max_batch, cfg.triton.opset)

    # end-to-end: same ranked lists with ONNX towers as with PyTorch towers
    onnx = Recommender(store, cfg, rec.two_tower, rec.gcn, rec.ranker, OnnxTowers(repo))
    sub = users[:200]
    a = rec.rank(T, sub, {k: v[:200] for k, v in uf.items()})
    T.cache.clear()
    b = onnx.rank(T, sub, {k: v[:200] for k, v in uf.items()})
    T.cache.clear()
    top_a = a.groupby("uidx").head(10)[["uidx", "jidx"]].to_numpy()
    top_b = b.groupby("uidx").head(10)[["uidx", "jidx"]].to_numpy()
    rep["e2e_top10_identical_share"] = float((top_a == top_b).all(1).mean()) if len(top_a) == len(top_b) else 0.0

    # CPU latency: user tower, batch 1 and 1024
    tt = rec.towers
    for bs in (1, 1024):
        f = {k: v[:bs] for k, v in uf.items()}
        for name, be in (("torch", tt), ("onnxruntime", onnx.towers)):
            be.encode_users(f)
            ts = []
            for _ in range(30 if bs == 1 else 5):
                t0 = time.perf_counter()
                be.encode_users(f)
                ts.append((time.perf_counter() - t0) * 1000)
            rep[f"user_tower_{name}_bs{bs}_ms_p50"] = round(float(np.median(ts)), 3)
    rep["onnx_bytes"] = {m: (repo / m / "1" / "model.onnx").stat().st_size for m in ("user_tower", "item_tower")}
    _update_results(cfg, "export", rep)
    if rep["user_tower_parity"]["max_abs_diff"] > 1e-4 or rep["item_tower_parity"]["max_abs_diff"] > 1e-4:
        raise SystemExit(f"ONNX parity failed: {rep}")
    log.info("export: %s", {k: v for k, v in rep.items() if "parity" in k or "ms" in k or "e2e" in k})


def stage_notify(cfg):
    from pathrank.notify.models import build_snapshots, fit_eval
    from pathrank.notify.policy import coverage_curve, sponsored_slot

    store = load_store(cfg)
    with tracking.run(cfg, "notify") as tr:
        with timer("snapshots"):
            train, test = build_snapshots(store, cfg)
        models, m = fit_eval(train, test, cfg.seed)
        m |= coverage_curve(models["scored_test"])
        ex = sponsored_slot(np.array([0.30, 0.22, 0.18, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03]),
                            sponsored_score=0.15, bid=2.0)
        m["sponsored_slot_example"] = ex
        tr.metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
        with open(cfg.art_dir / "notify_models.pkl", "wb") as f:
            pickle.dump({"clf": models["clf"], "reg": models["reg"]}, f)
        models["scored_test"].to_parquet(cfg.art_dir / "notify_scored_test.parquet")
    _update_results(cfg, "notify", m)
    log.info("notify: %s", {k: v for k, v in m.items() if not isinstance(v, dict)})


def stage_agent(cfg):
    """Notification agent eval, run through the evalwright SDK (sdk/python): every case is a traced
    `eval` span with the LangGraph nodes nested under it, results are stored as an experiment and
    compared case-by-case with the previous pipeline run, and quality gates are checked."""
    import evalwright as ew

    from pathrank.agent.graph import build_agent, run_agent
    from pathrank.agent.guardrails import check
    from pathrank.agent.llm import get_llm
    from pathrank.notify.policy import FrequencyCap
    from pathrank.serve.service import Service

    svc = Service.from_artifacts(cfg)
    llm = get_llm(cfg.agent.provider, cfg.agent.model, cfg.agent.openai_model)
    cap = FrequencyCap(cfg.notify.weekly_cap)
    users = svc.eval_users(cfg.agent.eval_users, cfg.seed + 7)
    user_ids = [int(u) for u in svc.store.users.user_id.to_numpy()[users]]
    svc.recommend(user_ids[0], 10)                     # warm per-context caches before concurrent cases

    agent = build_agent(svc.profile, lambda uid: svc.recommend(uid, 10)["recommendations"], cap.try_acquire, llm,
                        cfg.agent.max_retries)
    trace_file = cfg.art_dir / "traces" / "agent.jsonl"
    trace_file.unlink(missing_ok=True)
    tracer = ew.init("pathrank-agent", exporter=ew.JsonlExporter(trace_file), flush_interval=0.2)

    def task(user_id: int) -> dict:
        o = run_agent(agent, user_id)
        return {"status": o.get("status"), "attempts": o.get("attempts"), "errors": o.get("errors"),
                "judge": (o.get("judge") or {}).get("overall"), "message": o.get("message")}

    def delivered(output) -> float:
        """The message passed every guardrail and is ready to send."""
        return float(output["status"] == "ready")

    def first_try(output) -> float | None:
        """Ready without a guardrail-triggered redraft."""
        return None if output["status"] != "ready" else float(output["attempts"] == 1)

    def judge_quality(output) -> float | None:
        """LLM-judge (or template rubric) overall score, rescaled from 1-5 to 0-1."""
        return None if output["judge"] is None else (float(output["judge"]) - 1) / 4

    with timer("agent_eval"):
        res = ew.Eval("pathrank-agent", data=[{"input": u, "metadata": {"llm": llm.name}} for u in user_ids],
                      task=task, scores=[delivered, first_try, judge_quality], max_concurrency=cfg.agent.max_concurrency,
                      store_dir=cfg.art_dir / ".evalwright", metadata={"llm": llm.name, "model": cfg.agent.model})
    gate_failures = res.check({"delivered": cfg.agent.min_delivered}, cfg.agent.max_regression, 0.0)

    # frequency cap: after the eval run used each user's first send, two more runs must end capped
    with ew.span("frequency_cap_check", type="task"):
        for _ in range(cfg.notify.weekly_cap):
            last = [run_agent(agent, uid)["status"] for uid in user_ids[:5]]
    # adversarial check: does the guardrail catch hallucinated jobs, bad ids, invented pay/contact info?
    recs = svc.recommend(user_ids[0], 10)["recommendations"]
    bad = [{"subject": "New match", "body": 'Apply to "Senior Astronaut Pilot" today', "job_ids": [recs[0]["job_id"]]},
           {"subject": "New match", "body": f'Apply to "{recs[0]["title"]}"', "job_ids": [-1]},
           {"subject": "Pay!", "body": f'"{recs[0]["title"]}" pays $90k, call 512-555-0199', "job_ids": [recs[0]["job_id"]]}]
    caught = sum(bool(check(b, recs)) for b in bad)
    tracer.flush(10)
    n_spans = sum(1 for _ in open(trace_file)) if trace_file.exists() else 0

    outs = [c["output"] or {} for c in res.cases]
    status = pd.Series([o.get("status") for o in outs])
    attempts = pd.Series([o.get("attempts") for o in outs], dtype="float")
    m = {"llm_backend": llm.name, "n_users": len(res.cases), "ready_rate": float((status == "ready").mean()),
         "blocked_rate": float((status == "blocked").mean()), "mean_attempts": float(attempts.mean()),
         "judge_mean": res.score("judge_quality"),
         "frequency_cap_enforced": all(s == "capped" for s in last),
         "adversarial_caught": f"{caught}/{len(bad)}",
         "example_message": next((o.get("message") for o in outs if o.get("status") == "ready"), None),
         "evalwright": {
             "experiment_id": res.experiment_id,
             "scores": {k: {"mean": v["mean"], "ci95": v["ci95"], "n": v["n"]} for k, v in res.summary["scores"].items()},
             "latency_ms": res.summary["latency_ms"], "error_rate": res.summary["error_rate"],
             "comparison": None if not res.comparison else {
                 "baseline": res.comparison["baseline"],
                 "scores": {k: {"diff": v["diff"], "ci95": v["ci95"], "improved": v["improved"], "regressed": v["regressed"]}
                            for k, v in res.comparison["scores"].items()}},
             "gate_failures": gate_failures,
             "trace_file": str(trace_file), "n_spans": n_spans, "tracer_stats": tracer.stats()}}
    pd.DataFrame(res.cases).to_json(cfg.art_dir / "agent_eval.jsonl", orient="records", lines=True)
    with tracking.run(cfg, "agent") as tr:
        tr.metrics({k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        tr.metrics({f"eval_{k}": v["mean"] for k, v in m["evalwright"]["scores"].items() if v["mean"] is not None})
    ew.set_tracer(ew.Tracer(enabled=False))           # stop tracing for the rest of the pipeline
    _update_results(cfg, "agent", m)
    log.info("agent: %s", {k: v for k, v in m.items() if k not in ("example_message", "evalwright")})
    log.info("agent eval (evalwright):\n%s", res)


def stage_monitor(cfg):
    from pathrank.monitoring.drift import drift_table, evidently_report
    from pathrank.ranking.features import RANKER_V2

    d = cfg.art_dir
    ref = pd.read_parquet(d / "ref_features.parquet")
    cur = pd.read_parquet(d / "cur_features.parquet")
    tab = drift_table(ref, cur, RANKER_V2)
    tab.to_csv(d / "drift_psi.csv", index=False)
    ev = evidently_report(ref[RANKER_V2].sample(min(20_000, len(ref)), random_state=0),
                          cur[RANKER_V2].sample(min(20_000, len(cur)), random_state=0), str(d / "drift_report.html"))
    n_over = int((tab.psi > cfg.mlops.drift_psi_retrain).sum())
    flag = {"retrain_required": n_over > cfg.mlops.max_drifting_features, "features_over_threshold": n_over,
            "threshold_psi": cfg.mlops.drift_psi_retrain, "drifting": tab[tab.psi > cfg.mlops.drift_psi_retrain].feature.tolist()}
    save_json(flag, d / "retrain_flag.json")
    m = {"retrain_required": flag["retrain_required"],
         "n_drifting": int((tab.status == "drift").sum()), "n_watch": int((tab.status == "watch").sum()),
         "top_psi": tab.head(5)[["feature", "psi", "status"]].to_dict("records"), "evidently_html": ev is not None}
    if ev is not None:
        m["evidently"] = {k: ev[k] for k in ("version", "drifted_columns", "drifted_share") if k in ev}
        m["evidently"]["n_columns"] = len(ev["columns"])
        m["evidently"]["skipped_empty"] = ev.get("skipped_empty", [])
        m["evidently"]["top"] = sorted(ev["columns"].items(), key=lambda kv: -kv[1]["score"])[:5]
    _update_results(cfg, "monitor", m)
    log.info("monitor: %s", m)


def stage_report(cfg):
    from pathrank.analytics.report import build
    from scripts.write_results import write_results

    store = load_store(cfg)
    res = load_json(_results_path(cfg))
    scored = cfg.art_dir / "notify_scored_test.parquet"
    flat = {}
    if "rank" in res:
        flat["ranker_v2 ndcg@10"] = res["rank"]["test"]["ranker_v2"]["ndcg@10"]
        flat["popularity ndcg@10"] = res["rank"]["test"]["popularity"]["ndcg@10"]
    if "notify" in res:
        flat["apply classifier AUC"] = res["notify"]["clf_auc"]
        flat["return-time MAE (days)"] = res["notify"]["reg_mae_days"]
    build(store.apps, pd.read_parquet(scored) if scored.exists() else None, flat, str(cfg.art_dir / "analytics.html"))
    write_results(cfg, res, store.manifest)


STAGES = {"data": stage_data, "features": stage_features, "validate": stage_validate, "featurestore": stage_featurestore, "retrieval": stage_retrieval, "rank": stage_rank, "export": stage_export,
          "notify": stage_notify, "agent": stage_agent, "monitor": stage_monitor, "register": stage_register, "report": stage_report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/careerbuilder.yaml")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. ranker.eval_users=2000")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed_everything(cfg.seed)
    stages = ALL if a.stages == "all" else a.stages.split(",")
    for s in stages:
        log.info("===== stage: %s =====", s)
        with timer(f"stage_{s}"):
            STAGES[s](cfg)


if __name__ == "__main__":
    main()
