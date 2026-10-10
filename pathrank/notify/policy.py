"""Send decision, send time, frequency cap, and a sponsored-slot expected-value rule.

Offline we cannot measure causal uplift (no randomized sends in the data). What we CAN
measure honestly is targeting: how many sends it takes to reach a given share of the users
who go on to apply. That is what `coverage_curve` reports, and what the resume bullet
should say.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd


def coverage_curve(scored: pd.DataFrame, targets=(0.5, 0.8, 0.9)) -> dict:
    s = scored.sort_values("p_apply", ascending=False)
    y = s.y_apply.to_numpy()
    cum = np.cumsum(y) / max(y.sum(), 1)
    out = {}
    for tgt in targets:
        k = int(np.searchsorted(cum, tgt) + 1)
        frac = k / len(s)
        out[f"send_frac_for_{int(tgt * 100)}pct_applicants"] = frac
        out[f"send_reduction_vs_all_at_{int(tgt * 100)}pct"] = 1 - frac
        # random targeting needs `tgt` share of users to reach `tgt` share of applicants
        out[f"send_reduction_vs_random_at_{int(tgt * 100)}pct"] = 1 - frac / tgt
    return out


class FrequencyCap:
    """In-memory sliding-window cap (swap for Redis in prod)."""

    def __init__(self, per_week: int):
        self.per_week = per_week
        self.sent: dict[int, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allowed(self, user: int, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        with self._lock:
            q = self.sent[user]
            while q and now - q[0] > 7 * 86400:
                q.popleft()
            return len(q) < self.per_week

    def record(self, user: int, now: float | None = None) -> None:
        with self._lock:
            self.sent[user].append(now if now is not None else time.time())

    def try_acquire(self, user: int, now: float | None = None) -> bool:
        """Atomic check-and-record: concurrent requests can't both pass a full cap."""
        now = now if now is not None else time.time()
        with self._lock:
            q = self.sent[user]
            while q and now - q[0] > 7 * 86400:
                q.popleft()
            if len(q) >= self.per_week:
                return False
            q.append(now)
            return True


def decide(p_apply: float, pred_days: float, threshold: float, cap_ok: bool) -> dict:
    """Send if likely to engage and under cap; schedule the day before predicted return."""
    if not cap_ok:
        return {"send": False, "reason": "frequency_cap"}
    if p_apply < threshold:
        return {"send": False, "reason": "low_engagement_probability", "p_apply": p_apply}
    delay = int(np.clip(np.floor(pred_days) - 1, 0, 6))
    return {"send": True, "send_in_days": delay, "p_apply": p_apply, "pred_days_to_return": pred_days}


def sponsored_slot(organic_scores: np.ndarray, sponsored_score: float, bid: float,
                   slot: int = 2, value_per_apply: float = 1.0, alpha: float = 1.0) -> dict:
    """Insert a sponsored job at `slot` only if expected revenue beats expected relevance loss.

    Scores are treated as apply probabilities (calibrate the ranker before trusting this).
    loss = alpha * value_per_apply * (sum of exposure-weighted organic p before insertion - after).
    """
    w = 1 / np.log2(np.arange(2, len(organic_scores) + 3))
    before = (organic_scores * w[: len(organic_scores)]).sum()
    after_list = np.insert(organic_scores, slot, sponsored_score)[: len(organic_scores)]
    after = (after_list * w[: len(after_list)]).sum()
    loss = alpha * value_per_apply * (before - after)
    revenue = bid * sponsored_score * w[slot]
    return {"insert": bool(revenue > loss), "expected_revenue": float(revenue), "relevance_loss": float(loss)}
