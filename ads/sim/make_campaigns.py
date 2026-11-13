"""Build sponsored-job campaigns on real OLX jobs, plus the inputs the click simulator replays.

Writes three files to ads/data/:
  campaigns.json   what the ad server loads: adId, advertiserId, jobId, bidMicros, dailyBudgetMicros
  true_ctr.json    the hidden click-through rate per ad. ONLY the simulator reads this; the ad
                   server never sees it and has to learn CTRs from clicks through Flink.
  sim_inputs.json  real OLX user ids to replay, and the organic candidate pool with scores

Where the numbers come from:
  - Jobs, users and the organic pool are real (OLX events.parquet, the same file the Kafka
    producer replays).
  - OLX has no ads, so bids, budgets and advertisers are generated (seeded).
  - The hidden CTR is generated too, but anchored to real behaviour: jobs whose OLX interactions
    are mostly deeper actions (bookmark, chat, phone reveal) instead of plain clicks get a higher
    CTR. Formula: ctr = clip(0.005 + 0.08 * deep_share + noise, 0.003, 0.10).

  python ads/sim/make_campaigns.py --events data/raw/olx/events.parquet
  python ads/sim/make_campaigns.py --synthetic        # no Kaggle data needed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_events(path: str | None, synthetic: bool, seed: int) -> pd.DataFrame:
    if synthetic or not path or not Path(path).exists():
        if not synthetic:
            print(f"[make_campaigns] {path} not found, using synthetic OLX-shaped events")
        rng = np.random.default_rng(seed)
        n, n_users, n_jobs = 2_000_000, 200_000, 20_000
        item = rng.zipf(1.3, n) % n_jobs
        # each job gets its own deep-action rate, so jobs genuinely differ (like real OLX jobs do)
        deep_rate = rng.beta(2, 8, n_jobs)
        deep = rng.random(n) < deep_rate[item]
        ev = np.where(deep, rng.choice(["bookmark", "chat_click", "contact_phone_click_1"], n), "click")
        return pd.DataFrame({"user": rng.integers(0, n_users, n), "item": item, "event": ev})
    cols = ["user", "item", "event"]
    return pd.read_parquet(path, columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/raw/olx/events.parquet")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="ads/data")
    ap.add_argument("--n-ads", type=int, default=100,
                    help="fewer ads = more impressions per ad, so CTRs are learnable in a laptop-sized run")
    ap.add_argument("--n-advertisers", type=int, default=40)
    ap.add_argument("--min-events", type=int, default=200, help="only jobs with at least this many OLX events")
    ap.add_argument("--n-users", type=int, default=50_000, help="real user ids the simulator replays")
    ap.add_argument("--pool", type=int, default=300, help="organic candidate pool (most popular jobs)")
    ap.add_argument("--budget-scale", type=float, default=1.0,
                    help="multiply every daily budget; large values switch budgets off to isolate CTR learning")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = load_events(a.events, a.synthetic, a.seed)
    d["deep"] = d["event"].astype(str) != "click"
    per_job = d.groupby("item").agg(n=("deep", "size"), deep_share=("deep", "mean")).reset_index()
    per_job = per_job.sort_values("n", ascending=False)

    # organic pool = most popular jobs; score = popularity share scaled into an apply-probability range
    pool = per_job.head(a.pool).copy()
    pool["score"] = 0.005 + 0.06 * (pool["n"] / pool["n"].max())

    # sponsored jobs come from the next tier (popular enough to have a stable deep_share, but not
    # already the organic top results)
    eligible = per_job[(per_job["n"] >= a.min_events) & (~per_job["item"].isin(pool["item"]))]
    if len(eligible) < a.n_ads:
        eligible = per_job[~per_job["item"].isin(pool["item"])].head(a.n_ads * 2)
    picked = eligible.sample(n=min(a.n_ads, len(eligible)), random_state=a.seed).reset_index(drop=True)

    true_ctr = np.clip(0.005 + 0.08 * picked["deep_share"].to_numpy() + rng.normal(0, 0.004, len(picked)), 0.003, 0.10)
    bids = np.clip(rng.lognormal(mean=np.log(0.8), sigma=0.5, size=len(picked)), 0.10, 5.0)  # USD per click
    advertisers = rng.integers(0, a.n_advertisers, len(picked))
    adv_budget = {i: float(rng.choice([25, 50, 100, 250, 500])) * a.budget_scale for i in range(a.n_advertisers)}

    campaigns, hidden = [], {}
    for i, row in picked.iterrows():
        ad_id = f"ad-{i:04d}"
        adv = int(advertisers[i])
        campaigns.append({
            "adId": ad_id,
            "advertiserId": f"adv-{adv:03d}",
            "jobId": int(row["item"]),
            "bidMicros": int(round(bids[i] * 1_000_000)),
            "dailyBudgetMicros": int(adv_budget[adv] * 1_000_000),
        })
        hidden[ad_id] = round(float(true_ctr[i]), 6)

    users = d["user"].drop_duplicates()
    users = users.sample(n=min(a.n_users, len(users)), random_state=a.seed).astype(int).tolist()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "campaigns.json").write_text(json.dumps(campaigns, indent=1))
    (out / "true_ctr.json").write_text(json.dumps(hidden, indent=1))
    (out / "sim_inputs.json").write_text(json.dumps({
        "users": users,
        "pool": [{"jobId": int(j), "score": round(float(s), 6)} for j, s in zip(pool["item"], pool["score"])],
        "source": "synthetic" if (a.synthetic or not Path(a.events).exists()) else a.events,
    }))
    tc = np.array(list(hidden.values()))
    print(f"[make_campaigns] {len(campaigns)} ads, {a.n_advertisers} advertisers, {len(users):,} users, "
          f"pool {len(pool)} jobs; hidden CTR mean {tc.mean():.4f} (min {tc.min():.4f}, max {tc.max():.4f}) -> {out}/")


if __name__ == "__main__":
    main()
