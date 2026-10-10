"""Scale benchmark on 10M+ synthetic applications (CareerBuilder schema).

Measures throughput, not accuracy: generation, feature-store build, as-of context build,
two-tower and LightGCN training step rates on the full data, FAISS retrieval QPS, batch
ranking throughput, and single-user request latency (p50/p95) through the serving path.

  python scripts/benchmark_scale.py --config configs/synthetic_10m.yaml
"""
from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathrank.config import load_config  # noqa: E402
from pathrank.utils import log, save_json, seed_everything  # noqa: E402


def rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synthetic_10m.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--train-steps", type=int, default=50)
    ap.add_argument("--latency-users", type=int, default=50)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed_everything(cfg.seed)
    B: dict = {"machine": f"{platform.machine()} {platform.system()}", "python": platform.python_version()}

    def mark(name, t0):
        B[f"{name}_s"] = round(time.perf_counter() - t0, 2)
        B[f"{name}_peak_rss_gb"] = round(rss_gb(), 2)
        log.info("%s: %.1fs (peak RSS %.1f GB)", name, B[f"{name}_s"], rss_gb())

    # 1. data
    from pathrank.data.synthetic import generate
    raw = Path(cfg.data.raw_dir) / "synthetic"
    t0 = time.perf_counter()
    if not (raw / "apps.tsv").exists():
        B["dataset"] = generate(str(raw), cfg.data.synth_users, cfg.data.synth_jobs, cfg.data.synth_apps, cfg.seed)
        mark("generate", t0)
    else:
        B["generate_s"] = "cached"
    store_cached = (cfg.store_dir / "manifest.json").exists()

    # 2. feature store
    from pathrank.features.pipeline import AsOf, build_store, load_store
    t0 = time.perf_counter()
    build_store(cfg)
    if store_cached:
        B["build_feature_store_s"] = "cached"
    else:
        mark("build_feature_store", t0)
    t0 = time.perf_counter()
    store = load_store(cfg)
    mark("load_feature_store", t0)
    B["n_users"], B["n_jobs"], B["n_apps"] = len(store.users), len(store.jobs), len(store.apps)
    B["total_records"] = B["n_users"] + B["n_jobs"] + B["n_apps"]
    c = store.cutoffs
    hb = cfg.two_tower.hash_buckets

    t0 = time.perf_counter()
    T = AsOf(store, c["test"], c["end"] + pd.Timedelta(seconds=1), hb)
    mark("asof_context", t0)
    B["catalog_size"] = int(len(T.catalog))

    # 3. two-tower: step throughput on the full retriever-train pair set
    import torch
    import torch.nn.functional as F
    from pathrank.retrieval.two_tower import TwoTower, _to_t
    R = AsOf(store, c["ctx"], c["val"], hb)
    pairs = R.labels[["uidx", "jidx"]].to_numpy()
    B["two_tower_train_pairs"] = int(len(pairs))
    t0 = time.perf_counter()
    users, items = np.unique(pairs[:, 0]), np.unique(pairs[:, 1])
    uf, jf = R.user_features(users), R.job_features(items)
    mark("two_tower_featurize", t0)
    m = TwoTower(uf["text"].shape[1], cfg.two_tower)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    up, ip = np.searchsorted(users, pairs[:, 0]), np.searchsorted(items, pairs[:, 1])
    bs = cfg.two_tower.batch_size
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(a.train_steps):
        bi = rng.integers(0, len(pairs), bs)
        u, v = m.user(_to_t(uf, up[bi])), m.item(_to_t(jf, ip[bi]))
        loss = F.cross_entropy(u @ v.T / 0.05, torch.arange(bs))
        opt.zero_grad()
        loss.backward()
        opt.step()
    dt = time.perf_counter() - t0
    B["two_tower_pairs_per_s"] = round(a.train_steps * bs / dt)
    B["two_tower_epoch_est_min"] = round(len(pairs) / B["two_tower_pairs_per_s"] / 60, 1)
    log.info("two-tower: %s pairs/s", f"{B['two_tower_pairs_per_s']:,}")

    # 4. LightGCN: full-graph propagation step time (free two-tower training data first)
    import gc
    del uf, jf, R, pairs, up, ip, opt
    gc.collect()
    from pathrank.retrieval.lightgcn import LightGCN
    e = store.apps[store.apps.ts < c["val"]].drop_duplicates(["uidx", "jidx"])
    uu, ii = np.unique(e.uidx), np.unique(e.jidx)
    g = LightGCN(len(uu), len(ii), cfg.lightgcn.dim, cfg.lightgcn.layers)
    t0 = time.perf_counter()
    g.set_graph(np.searchsorted(uu, e.uidx.to_numpy()), np.searchsorted(ii, e.jidx.to_numpy()))
    mark("lightgcn_build_graph", t0)
    B["lightgcn_edges"] = int(len(e))
    gopt = torch.optim.Adam(g.parameters(), lr=1e-2)
    steps = 5
    t0 = time.perf_counter()
    for _ in range(steps):
        Ue, Ie = g.propagate()
        bi = rng.integers(0, len(e), cfg.lightgcn.batch_size)
        loss = -(Ue[np.searchsorted(uu, e.uidx.to_numpy()[bi])] * Ie[rng.integers(0, len(ii), len(bi))]).sum(-1).mean()
        gopt.zero_grad()
        loss.backward()
        gopt.step()
    B["lightgcn_step_s"] = round((time.perf_counter() - t0) / steps, 3)
    B["lightgcn_epoch_est_min"] = round(len(e) / cfg.lightgcn.batch_size * B["lightgcn_step_s"] / 60, 1)
    B["lightgcn_peak_rss_gb"] = round(rss_gb(), 2)
    del g, gopt, Ue, Ie, e
    gc.collect()

    # 5. retrieval QPS + batch ranking throughput through the real Recommender
    from pathrank.ranking.features import RANKER_V2, attach_labels, pair_features
    from pathrank.ranking.ranker import LGBRanker
    from pathrank.recommender import Recommender
    from pathrank.retrieval.index import CatalogIndex
    from pathrank.retrieval.two_tower import encode_items, encode_users
    m.eval()
    t0 = time.perf_counter()
    iv = encode_items(m, T.job_features(T.catalog))
    idx = CatalogIndex(iv, T.catalog)
    mark("encode_and_index_catalog", t0)
    ev = T.eval_users(5000, 0)
    tuf = T.user_features(ev)
    q = encode_users(m, tuf)
    t0 = time.perf_counter()
    idx.search(q, cfg.retrieval.k_two_tower)
    B["faiss_flat_qps"] = round(len(q) / (time.perf_counter() - t0))
    hn = CatalogIndex(iv, T.catalog, kind="hnsw")
    t0 = time.perf_counter()
    hn.search(q, cfg.retrieval.k_two_tower)
    B["faiss_hnsw_qps"] = round(len(q) / (time.perf_counter() - t0))
    log.info("FAISS QPS flat %s, hnsw %s over %s jobs", B["faiss_flat_qps"], B["faiss_hnsw_qps"], f"{len(T.catalog):,}")

    rec = Recommender(store, cfg, m, None)
    T.cache["tt_index"] = idx
    sub = ev[:1000]
    t0 = time.perf_counter()
    cand, _ = rec.candidates(T, sub, {k: v[:1000] for k, v in tuf.items()})
    feats = attach_labels(T, pair_features(T, cand, {k: v[:1000] for k, v in tuf.items()}))
    dt = time.perf_counter() - t0
    B["batch_candidates_plus_features_users_per_s"] = round(len(sub) / dt, 1)
    B["candidate_rows_per_user"] = round(len(feats) / len(sub))
    rk = LGBRanker(RANKER_V2, cfg.ranker, 0)
    rk.cfg.n_estimators, rk.cfg.early_stopping = 100, 20
    t0 = time.perf_counter()
    rk.fit(feats)
    mark("ranker_fit_1k_users", t0)
    rec.ranker = rk
    t0 = time.perf_counter()
    rk.predict(feats)
    B["ranker_rows_per_s"] = round(len(feats) / (time.perf_counter() - t0))

    # 6. single-user latency through the serving path (features + 3 retrievers + rank)
    lat = []
    for u in ev[: a.latency_users]:
        t0 = time.perf_counter()
        rec.rank(T, np.array([u]))
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat[1:]) * 1000
    B["request_latency_ms_p50"] = round(float(np.percentile(lat, 50)), 1)
    B["request_latency_ms_p95"] = round(float(np.percentile(lat, 95)), 1)
    B["peak_rss_gb"] = round(rss_gb(), 2)
    out = cfg.art_dir / "benchmark.json"
    save_json(B, out)
    log.info("benchmark written to %s", out)
    for k, v in B.items():
        print(f"{k:45s} {v}")


if __name__ == "__main__":
    main()
