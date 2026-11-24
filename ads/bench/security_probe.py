"""Throw every kind of unauthorized request at a running, secured ad server and count how many are
turned away with the right status. Also checks that the legitimate versions of the same requests
succeed, so a server that rejects everything can't score well.

  python ads/bench/security_probe.py   (server on :8090 with ads.security.enabled=true, default demo users)

Categories (each against every relevant endpoint / campaign):
  no_token, garbage_token, tampered_signature, expired_token, wrong_issuer  -> 401
  wrong_role (advertiser calling auction/click/stats/admin, service calling the console API) -> 403
  cross_tenant (advertiser reading another advertiser's campaign, stats or spend) -> 403
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
from collections import Counter, defaultdict

import httpx

DEV_SECRET = "q5lJ0m8iXH4h6gQ1xYw2ZbW9cT3rN7uV0kE5sA8dF1gH2jK4lP6oI9uY3tR5eW7q"  # application.yml default


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def forge(secret_b64: str, claims: dict) -> str:
    """An HS256 token signed with the real key, to build expired / wrong-issuer tokens."""
    key = base64.b64decode(secret_b64)
    head = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64url(json.dumps(claims).encode())
    sig = hmac.new(key, f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}.{b64url(sig)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8090")
    ap.add_argument("--campaigns", default="ads/data/campaigns.json")
    ap.add_argument("--advertisers", default="adv-000,adv-001,adv-002", help="advertiser demo users (password <id>-pass)")
    ap.add_argument("--secret", default=os.environ.get("ADS_SECURITY_JWT_SECRET", DEV_SECRET))
    ap.add_argument("--out", default="results/ads/security.json")
    a = ap.parse_args()

    c = httpx.Client(base_url=a.base, timeout=10)

    def login(user: str, pw: str) -> str:
        r = c.post("/auth/token", json={"username": user, "password": pw})
        r.raise_for_status()
        return r.json()["accessToken"]

    service, admin = login("jobboard", "jobboard-pass"), login("admin", "admin-pass")
    advs = {adv: login(adv, f"{adv}-pass") for adv in a.advertisers.split(",")}
    campaigns = json.load(open(a.campaigns))
    owner = {x["adId"]: x["advertiserId"] for x in campaigns}

    now = int(time.time())
    bad_tokens = {
        "garbage_token": "not.a.jwt",
        "tampered_signature": service[:-2] + ("AA" if not service.endswith("AA") else "BB"),
        "expired_token": forge(a.secret, {"iss": "pathrank-ads", "sub": "x", "iat": now - 7200, "exp": now - 3600,
                                          "roles": ["ADMIN"]}),
        "wrong_issuer": forge(a.secret, {"iss": "evil", "sub": "x", "iat": now, "exp": now + 3600, "roles": ["ADMIN"]}),
    }
    auction = {"userId": 1, "organic": [{"jobId": 1, "score": 0.1}]}
    endpoints = [("POST", "/ads/auction", auction), ("POST", "/ads/click", {"impressionId": "x"}),
                 ("GET", "/ads/stats", None), ("GET", "/api/campaigns", None), ("GET", "/admin/budget/adv-000", None)]

    cases: list[tuple[str, str, str, dict | None, str | None, int]] = []  # category, method, path, body, token, want
    for m, p, b in endpoints:
        cases.append(("no_token", m, p, b, None, 401))
        for cat, tok in bad_tokens.items():
            cases.append((cat, m, p, b, tok, 401))
    for adv, tok in advs.items():
        for m, p, b in endpoints:
            if p.startswith("/api"):
                continue
            cases.append(("wrong_role", m, p, b, tok, 403))
    cases.append(("wrong_role", "GET", "/api/campaigns", None, service, 403))
    for adv, tok in advs.items():
        for ad, own in owner.items():
            if own == adv:
                continue
            cases.append(("cross_tenant", "GET", f"/api/campaigns/{ad}", None, tok, 403))
            cases.append(("cross_tenant", "GET", f"/api/campaigns/{ad}/stats", None, tok, 403))
        for other in sorted(set(owner.values()) - {adv})[:10]:
            cases.append(("cross_tenant", "GET", f"/api/advertisers/{other}/spend", None, tok, 403))

    allowed = [("allowed", "POST", "/ads/auction", auction, service, 200),
               ("allowed", "GET", "/ads/stats", None, admin, 200)]
    for adv, tok in advs.items():
        for ad, own in owner.items():
            if own == adv:
                allowed.append(("allowed", "GET", f"/api/campaigns/{ad}", None, tok, 200))

    got: dict[str, Counter] = defaultdict(Counter)
    wrong: list[str] = []
    for cat, m, p, b, tok, want in cases + allowed:
        h = {"Authorization": f"Bearer {tok}"} if tok else {}
        r = c.request(m, p, json=b, headers=h)
        ok = r.status_code == want
        got[cat]["ok" if ok else "wrong"] += 1
        if not ok:
            wrong.append(f"{cat} {m} {p}: want {want}, got {r.status_code}")

    unauthorized = sum(v["ok"] + v["wrong"] for k, v in got.items() if k != "allowed")
    rejected = sum(v["ok"] for k, v in got.items() if k != "allowed")
    out = {
        "unauthorized_requests": unauthorized,
        "rejected_correctly": rejected,
        "allowed_requests": got["allowed"]["ok"] + got["allowed"]["wrong"],
        "allowed_ok": got["allowed"]["ok"],
        "by_category": {k: dict(v) for k, v in sorted(got.items())},
        "mismatches": wrong[:50],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "mismatches"}, indent=2))
    if wrong:
        print(f"{len(wrong)} mismatches, first: {wrong[0]}")


if __name__ == "__main__":
    main()
