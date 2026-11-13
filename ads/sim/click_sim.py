"""Replay real OLX users against the ad server and click with a hidden per-ad CTR.

For each request: pick a real OLX user id, sample 10 organic jobs from the popular pool (sorted by
score), POST /ads/auction, then for every sponsored row click with probability
    true_ctr[ad] * position_bias[position],   position_bias = 1 / log2(position + 2), 1.0 at position 0
and POST /ads/click.

The server only learns CTRs from those clicks (ad server -> Kafka -> Flink -> Redis -> ad server).
At the end this compares what it learned to the hidden truth and writes results/ads/sim.json:
  - learned vs expected CTR per ad (MAE, Spearman rank correlation)
  - share of impressions going to the best ads (top-quartile hidden CTR), first 10% of requests
    vs last 10%: if the feedback loop works, the auction shifts traffic to better ads over time
  - overall CTR first 10% vs last 10%

  python ads/sim/click_sim.py --requests 200000 --concurrency 16
"""
from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8090")
    ap.add_argument("--data", default="ads/data")
    ap.add_argument("--requests", type=int, default=200_000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--organic", type=int, default=10)
    ap.add_argument("--settle", type=float, default=12.0, help="seconds to wait for Flink to flush the last windows")
    ap.add_argument("--out", default="results/ads/sim.json")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    data = Path(a.data)
    inputs = json.loads((data / "sim_inputs.json").read_text())
    true_ctr: dict[str, float] = json.loads((data / "true_ctr.json").read_text())
    users, pool = inputs["users"], inputs["pool"]
    q75 = float(np.quantile(list(true_ctr.values()), 0.75))
    best = {k for k, v in true_ctr.items() if v >= q75}

    n = a.requests
    log = [None] * n  # per request: (n_sponsored, n_best, n_clicks)
    expected_clicks: dict[str, float] = {}
    lock = threading.Lock()
    local = threading.local()
    errors = [0]

    def client() -> httpx.Client:
        if not hasattr(local, "c"):
            local.c = httpx.Client(base_url=a.base, timeout=5.0)
            local.rng = random.Random(a.seed + threading.get_ident())
        return local.c

    def one(i: int) -> None:
        c = client()
        rng: random.Random = local.rng
        organic = sorted(rng.sample(pool, a.organic), key=lambda o: -o["score"])
        try:
            r = c.post("/ads/auction", json={"userId": rng.choice(users), "organic": organic})
            r.raise_for_status()
        except httpx.HTTPError:
            with lock:
                errors[0] += 1
            return
        n_sp = n_best = n_clk = 0
        for it in r.json()["items"]:
            if not it["sponsored"]:
                continue
            ad, pos = it["adId"], it["rank"]
            p = true_ctr.get(ad, 0.0) * (1.0 / math.log2(pos + 2))
            n_sp += 1
            n_best += ad in best
            with lock:
                expected_clicks[ad] = expected_clicks.get(ad, 0.0) + p
            if rng.random() < p:
                try:
                    c.post("/ads/click", json={"impressionId": it["impressionId"]})
                    n_clk += 1
                except httpx.HTTPError:
                    with lock:
                        errors[0] += 1
        log[i] = (n_sp, n_best, n_clk)

    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        for k, _ in enumerate(ex.map(one, range(n))):
            if (k + 1) % 20_000 == 0:
                print(f"[click_sim] {k + 1:,}/{n:,} requests, {(k + 1) / (time.time() - t0):,.0f} req/s")
    wall = time.time() - t0
    print(f"[click_sim] done in {wall:.1f}s; waiting {a.settle}s for Flink to flush the last windows")
    time.sleep(a.settle)

    stats = httpx.get(f"{a.base}/ads/stats", timeout=10).json()
    learned, expected, truth = [], [], []
    abs_err = []
    for ad, s in stats["ads"].items():
        if s["served_impressions"] < 200 or ad not in true_ctr:
            continue  # too few impressions to judge
        exp_ctr = expected_clicks.get(ad, 0.0) / s["served_impressions"]
        flink_ctr = s["flink_clicks"] / s["flink_impressions"] if s["flink_impressions"] else 0.0
        learned.append(s["pctr"])
        expected.append(exp_ctr)
        truth.append(true_ctr[ad])
        abs_err.append(abs(flink_ctr - exp_ctr))

    def window(rows):
        rows = [r for r in rows if r]
        sp = sum(r[0] for r in rows)
        return {"requests": len(rows), "impressions": sp,
                "best_ad_share": (sum(r[1] for r in rows) / sp) if sp else 0.0,
                "ctr": (sum(r[2] for r in rows) / sp) if sp else 0.0}

    tenth = max(1, n // 10)
    res = {
        "requests": n,
        "wall_s": round(wall, 2),
        "sim_req_per_s": round(n / wall, 1),
        "http_errors": errors[0],
        "ads_judged": len(learned),
        "flink_ctr_mae_vs_expected": float(np.mean(abs_err)) if abs_err else None,
        "expected_ctr_mean": float(np.mean(expected)) if expected else None,
        "spearman_pctr_vs_hidden_ctr": spearman(learned, truth),
        "first_10pct": window(log[:tenth]),
        "last_10pct": window(log[-tenth:]),
        "random_best_share": round(len(best) / len(true_ctr), 4),
        "server_totals": stats["totals"],
        "inputs_source": inputs.get("source"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "server_totals"}, indent=2))


if __name__ == "__main__":
    main()
