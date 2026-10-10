"""The 3-stage recommender: retrieve (4 sources) -> rank (LightGBM) -> fairness re-rank.

One class serves both offline evaluation (batches of thousands of users) and the API
(one user at a time), so features are computed by the same code in both places.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import evalwright as ew
import torch

from .features.pipeline import AsOf, Store
from .ranking.features import merge_candidates, pair_features
from .retrieval.index import CatalogIndex
from .retrieval.simple import location_popularity, text_retrieve
from .retrieval.two_tower import TwoTower


class Recommender:
    def __init__(self, store: Store, cfg, two_tower: TwoTower | None = None, gcn=None, ranker=None,
                 towers=None):
        from .inference.backends import TorchTowers

        self.s, self.cfg = store, cfg
        self.two_tower, self.gcn, self.ranker = two_tower, gcn, ranker
        # tower inference backend: torch (default) | onnx | triton  (see inference/backends.py)
        self.towers = towers or (TorchTowers(two_tower) if two_tower is not None else None)

    @property
    def tower_backend(self) -> str | None:
        return getattr(self.towers, "name", None)

    # ---- stage 1 ---------------------------------------------------------
    def _tt_catalog(self, ctx: AsOf) -> CatalogIndex:
        if "tt_index" not in ctx.cache:
            vecs = self.towers.encode_items(ctx.job_features(ctx.catalog))
            ctx.cache["tt_index"] = CatalogIndex(vecs, ctx.catalog)
        return ctx.cache["tt_index"]

    def candidates(self, ctx: AsOf, uidx: np.ndarray, uf: dict | None = None,
                   sources=("tt", "txt", "gcn", "lp")) -> tuple[pd.DataFrame, dict]:
        r = self.cfg.retrieval
        uf = uf if uf is not None else ctx.user_features(uidx)
        out: dict = {}
        if "tt" in sources and self.towers is not None:
            out["tt"] = self._tt_catalog(ctx).search(self.towers.encode_users(uf), r.k_two_tower)
        if "txt" in sources:
            out["txt"] = text_retrieve(ctx, uf["text"], r.k_text)
        if "gcn" in sources and self.gcn is not None:
            S, J, _ = self.gcn.retrieve(uidx, ctx.catalog, r.k_gcn)
            out["gcn"] = (S, J)
        if "lp" in sources:
            out["lp"] = location_popularity(ctx, uidx, r.k_locpop)
        return merge_candidates(uidx, out), out

    # ---- stage 2 ---------------------------------------------------------
    def rank(self, ctx: AsOf, uidx: np.ndarray, uf: dict | None = None, ranker=None) -> pd.DataFrame:
        # evalwright spans: no-ops unless tracing is initialized (API with EVALWRIGHT_EXPORT set)
        with ew.span("features.user", type="retrieval") as s:
            uf = uf if uf is not None else ctx.user_features(uidx)
            s.log(metrics={"users": len(uidx)})
        with ew.span("retrieve.candidates", type="retrieval") as s:
            cand, _ = self.candidates(ctx, uidx, uf)
            s.log(metrics={"candidates": len(cand)}, metadata={"towers": self.tower_backend})
        with ew.span("features.pairs", type="function"):
            feats = pair_features(ctx, cand, uf)
        with ew.span("rank.lightgbm", type="function") as s:
            rk = ranker or self.ranker
            feats["score"] = rk.predict(feats)
            s.log(metrics={"rows": len(feats)})
        return feats.sort_values(["uidx", "score"], ascending=[True, False], kind="stable")

    # ---- persistence -------------------------------------------------------
    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.two_tower.state_dict(), d / "two_tower.pt")
        with open(d / "gcn.pkl", "wb") as f:
            pickle.dump(self.gcn, f)
        with open(d / "ranker.pkl", "wb") as f:
            pickle.dump(self.ranker, f)

    @classmethod
    def load(cls, store: Store, cfg, d: Path, backend: str | None = None) -> "Recommender":
        """backend: torch | onnx | triton. Default: triton if cfg.triton.url / TRITON_URL is set."""
        import os

        from .inference.backends import TritonRanker, make_towers

        tt = TwoTower(store.job_vec.shape[1], cfg.two_tower)
        tt.load_state_dict(torch.load(d / "two_tower.pt", weights_only=True))
        tt.eval()
        gcn = pickle.load(open(d / "gcn.pkl", "rb")) if (d / "gcn.pkl").exists() else None
        ranker = pickle.load(open(d / "ranker.pkl", "rb"))
        url = os.environ.get("TRITON_URL", cfg.triton.url)
        backend = backend or os.environ.get("TOWER_BACKEND") or ("triton" if url else "torch")
        towers = make_towers(backend, tt, d / "model_repository", url, cfg.triton.max_batch)
        if backend == "triton":
            ranker_local = ranker
            ranker = TritonRanker(url, ranker_local.features)
            ranker.model = ranker_local.model      # SHAP reasons still run locally on the same trees
        return cls(store, cfg, tt, gcn, ranker, towers)
