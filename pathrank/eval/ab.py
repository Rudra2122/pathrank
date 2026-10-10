"""Replayed offline A/B test between two rankers on the held-out window.

Users are hash-split into arms the way an online test would (arm = hash(user) % 2), each arm
is scored with its own ranker, and the difference gets a bootstrap CI. Because this is
offline we also have the counterfactual for every user, so we report the paired estimate
too; the paired CI is much tighter, which is the point of showing both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def replay_ab(pu_a: pd.DataFrame, pu_b: pd.DataFrame, metric: str = "ndcg@10", seed: int = 0) -> dict:
    a = pu_a.set_index("uidx")[metric]
    b = pu_b.set_index("uidx")[metric].reindex(a.index)
    users = a.index.to_numpy()
    arm = (users * 2654435761 % 2**32) % 2  # Knuth multiplicative hash
    xa, xb = a.to_numpy()[arm == 0], b.to_numpy()[arm == 1]
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(xb, len(xb)).mean() - rng.choice(xa, len(xa)).mean() for _ in range(2000)]
    paired = (b - a).to_numpy()
    lo, hi = _boot_ci(paired, seed=seed)
    p_two = 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean())
    return {
        "metric": metric,
        "split_arm_A_mean": float(xa.mean()), "split_arm_B_mean": float(xb.mean()),
        "split_diff": float(xb.mean() - xa.mean()),
        "split_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "split_p_value": float(min(1.0, p_two)),
        "paired_diff": float(paired.mean()), "paired_ci95": [lo, hi],
        "paired_rel_lift": float(paired.mean() / max(a.mean(), 1e-9)),
        "n_users_A": int((arm == 0).sum()), "n_users_B": int((arm == 1).sum()),
    }
