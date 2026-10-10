"""Online service object: loads artifacts once, answers per-user requests.

Uses the same AsOf + Recommender code as the offline pipeline. "Now" is the dataset's test
cutoff by default (the data is from 2012), set SERVE_CUTOFF=end to serve as-of the last day.
"""
from __future__ import annotations

import os
import pickle

import evalwright as ew
import numpy as np
import pandas as pd

from ..features.pipeline import AsOf, load_store
from ..notify.models import NOTIFY_FEATURES, snapshot
from ..notify.policy import FrequencyCap, decide
from ..ranking.explain import reasons
from ..recommender import Recommender


class Service:
    def __init__(self, cfg, store, rec: Recommender, notify_models: dict | None, cutoff: str = "test"):
        self.cfg, self.store, self.rec = cfg, store, rec
        t = store.cutoffs[cutoff] if cutoff != "end" else store.cutoffs["end"] + pd.Timedelta(seconds=1)
        self._t_end = store.cutoffs["end"] + pd.Timedelta(seconds=1) if cutoff == "test" else None
        self._eval_ctx = None
        fc = cfg.featurestore
        repo = cfg.art_dir / "feature_repo" / "registry.db"
        if fc.enabled and fc.serve_from_online and cutoff == "test" and repo.exists():
            from ..featurestore.online import OnlineContext
            self.ctx = OnlineContext(store, cfg, t)       # features served from the online store
            self.feature_source = "feast_online"
        else:
            self.ctx = AsOf(store, t, self._t_end, cfg.two_tower.hash_buckets)
            self.feature_source = "asof_inprocess"
        self.uidx_of = pd.Series(store.users.uidx.to_numpy(), index=store.users.user_id.to_numpy())
        self.notify = notify_models
        self.cap = FrequencyCap(cfg.notify.weekly_cap)
        self._snap = None

    @classmethod
    def from_artifacts(cls, cfg) -> "Service":
        store = load_store(cfg)
        rec = Recommender.load(store, cfg, cfg.art_dir)
        if os.environ.get("MODEL_SOURCE") == "registry":
            # serve whatever the registry says is champion (promotion/rollback = moving an alias)
            import mlflow
            import mlflow.lightgbm
            mlflow.set_tracking_uri(cfg.mlflow_uri)
            name = f"{cfg.mlops.registry_prefix}-{cfg.data.source}-ranker"
            rec.ranker.model = mlflow.lightgbm.load_model(f"models:/{name}@champion")
            rec.ranker.model.best_iteration = 0       # registry copy is already truncated to best_iteration
        p = cfg.art_dir / "notify_models.pkl"
        nm = pickle.load(open(p, "rb")) if p.exists() else None
        return cls(cfg, store, rec, nm, os.environ.get("SERVE_CUTOFF", "test"))

    def eval_users(self, n: int, seed: int) -> np.ndarray:
        """Users with applications after the serving cutoff (for offline agent eval / tests)."""
        if self._eval_ctx is None:
            self._eval_ctx = AsOf(self.store, self.ctx.t, self._t_end, self.cfg.two_tower.hash_buckets)
        return self._eval_ctx.eval_users(n, seed)

    def online_features(self, user_id: int) -> dict:
        ui = np.array([self._uidx(user_id)])
        f = self.ctx.user_features(ui)
        return {"user_id": user_id, "source": self.feature_source,
                "n_apps": float(self.ctx.user_n_apps[ui[0]]),
                "days_since_last": None if np.isnan(self.ctx.user_last[ui[0]]) else float(self.ctx.user_last[ui[0]]),
                "history_token_count": int((f["tok"][0] > 0).sum()),
                "profile_embedding_dim": int(f["text"].shape[1]), "num": f["num"][0].round(4).tolist()}

    def _uidx(self, user_id: int) -> int:
        if user_id not in self.uidx_of.index:
            raise KeyError(user_id)
        return int(self.uidx_of[user_id])

    def profile(self, user_id: int) -> dict:
        u = self.store.users.iloc[self._uidx(user_id)]
        return {"user_id": int(user_id), "city": u.city, "state": u.state, "degree": u.degree,
                "major": u.major, "years_experience": float(u.years_exp),
                "past_titles": u.work_titles[:200]}

    @ew.traced(type="task", name="recommend")
    def recommend(self, user_id: int, k: int = 10) -> dict:
        ui = np.array([self._uidx(user_id)])
        ranked = self.rec.rank(self.ctx, ui).head(k).copy()
        if len(ranked) == 0:
            return {"user_id": user_id, "recommendations": []}
        ranked["reason"] = reasons(self.rec.ranker, ranked)
        J = self.store.jobs
        j = ranked.jidx.to_numpy()
        recs = [{"job_id": int(J.job_id.iat[x]), "title": J.title.iat[x], "city": J.city.iat[x],
                 "state": J.state.iat[x], "score": float(s), "reason": r}
                for x, s, r in zip(j, ranked.score.to_numpy(), ranked.reason)]
        return {"user_id": user_id, "as_of": str(self.ctx.t), "recommendations": recs}

    def notify_decision(self, user_id: int, threshold: float = 0.3) -> dict:
        if self.notify is None:
            return {"send": False, "reason": "notify models not trained"}
        ui = self._uidx(user_id)
        if self._snap is None:
            n = self.cfg.notify
            self._snap = snapshot(self.store, self.ctx.t, n.horizon_days, n.return_cap_days).set_index("uidx")
        if ui not in self._snap.index:
            return {"send": False, "reason": "no_history"}
        x = self._snap.loc[[ui], NOTIFY_FEATURES]
        p = float(self.notify["clf"].predict_proba(x)[0, 1])
        d = float(self.notify["reg"].predict(x)[0])
        if p < threshold:                                  # don't spend cap on a send we won't make
            return decide(p, d, threshold, True)
        ok = self.cap.try_acquire(user_id)                 # atomic: concurrent requests can't double-send
        return decide(p, d, threshold, ok)
