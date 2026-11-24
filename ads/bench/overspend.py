"""Two ad server processes, one shared budget: can they overspend it?

Starts two `ad-server.jar` JVMs (ports 8091, 8092) with ads.budget-store=jpa against PostgreSQL or
CockroachDB, then fires concurrent charges for ONE advertiser at both of them through the admin
endpoint (the same BudgetStore.charge() a real click runs). The budget fits exactly
BUDGET/PRICE clicks; we send ATTEMPTS. Afterwards: how many were charged, what the row says,
how many optimistic-lock / serialization retries it took, and charge latency.

  make ads-db-up && make ads-build
  python ads/bench/overspend.py --db postgres
  python ads/bench/overspend.py --db cockroach
-> results/ads/overspend_<db>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

PORTS = (8091, 8092)


def wait_up(port: int, proc: subprocess.Popen, log: str) -> None:
    for _ in range(90):
        if proc.poll() is not None:
            raise SystemExit(f"ad server on :{port} exited, see {log}")
        try:
            if httpx.get(f"http://localhost:{port}/actuator/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise SystemExit(f"ad server on :{port} did not come up, see {log}")


def retries(port: int) -> float:
    text = httpx.get(f"http://localhost:{port}/actuator/prometheus", timeout=5).text
    m = re.search(r"^ads_budget_retries_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", text, re.M)
    return float(m.group(1)) if m else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=["postgres", "cockroach"], required=True)
    ap.add_argument("--jar", default="ads/ad-server/target/ad-server.jar")
    ap.add_argument("--budget-micros", type=int, default=5_000_000)   # $5.00
    ap.add_argument("--price-micros", type=int, default=1_000)        # $0.001 -> 5,000 clicks fit
    ap.add_argument("--attempts", type=int, default=10_000)
    ap.add_argument("--threads", type=int, default=32, help="client threads, split across the two servers")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or f"results/ads/overspend_{a.db}.json"
    os.makedirs("logs", exist_ok=True)

    procs = []
    try:
        # one at a time: the second starts after the first has created the tables and seeded campaigns
        for port in PORTS:
            log = f"logs/ad-server-{a.db}-{port}.log"
            p = subprocess.Popen(
                ["java", "-jar", a.jar, f"--server.port={port}", f"--spring.profiles.active={a.db}",
                 "--ads.budget-store=jpa", "--ads.security.enabled=true"],
                stdout=open(log, "w"), stderr=subprocess.STDOUT)
            procs.append((port, p, log))
            wait_up(port, p, log)
            time.sleep(3)   # health is up before ApplicationRunners finish; make sure it stays up
            if p.poll() is not None:
                raise SystemExit(f"ad server on :{port} exited right after starting, see {log}")

        tokens = {}
        for port in PORTS:
            r = httpx.post(f"http://localhost:{port}/auth/token",
                           json={"username": "admin", "password": "admin-pass"}, timeout=10)
            r.raise_for_status()
            tokens[port] = r.json()["accessToken"]
        r0 = {port: retries(port) for port in PORTS}

        advertiser = f"adv-race-{a.db}-{int(time.time())}"
        local = threading.local()
        lat: list[float] = []
        charged = [0]
        errors = [0]
        lock = threading.Lock()

        def one(i: int) -> None:
            port = PORTS[i % 2]
            if not hasattr(local, "c"):
                local.c = {p: httpx.Client(base_url=f"http://localhost:{p}", timeout=30,
                                           headers={"Authorization": f"Bearer {tokens[p]}"}) for p in PORTS}
            t = time.perf_counter()
            try:
                r = local.c[port].post("/admin/budget/charge", json={
                    "advertiserId": advertiser, "dailyBudgetMicros": a.budget_micros, "priceMicros": a.price_micros})
                ok = r.status_code == 200 and r.json()["charged"]
                err = r.status_code != 200
            except httpx.HTTPError:
                ok, err = False, True
            dt = (time.perf_counter() - t) * 1000
            with lock:
                lat.append(dt)
                charged[0] += ok
                errors[0] += err

        t0 = time.perf_counter()
        with ThreadPoolExecutor(a.threads) as ex:
            list(ex.map(one, range(a.attempts)))
        wall = time.perf_counter() - t0
        for port, p, log in procs:
            if p.poll() is not None:
                raise SystemExit(f"ad server on :{port} died during the run, see {log}; results not written")

        spent = httpx.get(f"http://localhost:{PORTS[0]}/admin/budget/{advertiser}", timeout=10,
                          headers={"Authorization": f"Bearer {tokens[PORTS[0]]}"}).json()["spentMicros"]
        total_retries = sum(retries(p) - r0[p] for p in PORTS)
        fits = a.budget_micros // a.price_micros
        l = np.array(lat)
        out = {
            "db": a.db,
            "instances": len(PORTS),
            "client_threads": a.threads,
            "attempts": a.attempts,
            "clicks_that_fit_budget": int(fits),
            "charged": int(charged[0]),
            "spent_micros": int(spent),
            "budget_micros": a.budget_micros,
            "overspend_micros": max(0, int(spent) - a.budget_micros),
            "exact": int(charged[0]) == fits and int(spent) == a.budget_micros,
            "http_errors": int(errors[0]),
            "retries": int(total_retries),
            "retries_per_charged_click": round(total_retries / max(1, charged[0]), 3),
            "charges_per_s": round(a.attempts / wall, 1),
            "latency_ms": {"p50": round(float(np.percentile(l, 50)), 2), "p95": round(float(np.percentile(l, 95)), 2),
                           "p99": round(float(np.percentile(l, 99)), 2)},
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(out, open(out_path, "w"), indent=2)
        print(json.dumps(out, indent=2))
    finally:
        for _, p, _ in procs:
            p.terminate()
        for _, p, _ in procs:
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
