"""Stage 2: LightGBM LambdaRank over merged candidates."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..utils import log


class LGBRanker:
    def __init__(self, features: list[str], cfg, seed: int = 0, name: str = "ranker"):
        self.features, self.cfg, self.seed, self.name = list(features), cfg, seed, name
        self.model: lgb.Booster | None = None

    def fit(self, df: pd.DataFrame) -> dict:
        # lambdarank needs at least one positive per group to learn anything
        has_pos = df.groupby("uidx")["label"].transform("max") > 0
        df = df[has_pos].sort_values("uidx", kind="stable")
        users = df.uidx.unique()
        rng = np.random.default_rng(self.seed)
        va_users = set(rng.choice(users, max(1, len(users) // 10), replace=False))
        va = df.uidx.isin(va_users).to_numpy()
        tr_df, va_df = df[~va], df[va]
        dtr = lgb.Dataset(tr_df[self.features], tr_df["label"], group=tr_df.groupby("uidx", sort=False).size().to_numpy())
        dva = lgb.Dataset(va_df[self.features], va_df["label"], group=va_df.groupby("uidx", sort=False).size().to_numpy())
        params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10],
                  "num_leaves": self.cfg.num_leaves, "learning_rate": self.cfg.learning_rate,
                  "min_data_in_leaf": 50, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "lambdarank_truncation_level": 20, "verbose": -1, "seed": self.seed, "num_threads": 0}
        evals: dict = {}
        self.model = lgb.train(params, dtr, self.cfg.n_estimators, valid_sets=[dva], valid_names=["val"],
                               callbacks=[lgb.early_stopping(self.cfg.early_stopping, verbose=False),
                                          lgb.record_evaluation(evals)])
        best = self.model.best_iteration
        info = {"train_groups": int(tr_df.uidx.nunique()), "val_groups": int(va_df.uidx.nunique()),
                "train_rows": int(len(tr_df)), "best_iter": int(best),
                "val_ndcg@10_inner": float(evals["val"]["ndcg@10"][best - 1])}
        log.info("%s trained: %s", self.name, info)
        return info

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features], num_iteration=self.model.best_iteration)

    def importance(self) -> pd.Series:
        imp = self.model.feature_importance("gain")
        return pd.Series(imp / imp.sum(), index=self.features).sort_values(ascending=False)
