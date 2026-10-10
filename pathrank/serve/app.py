"""FastAPI app.  uvicorn pathrank.serve.app:app --port 8080

GET  /health
GET  /recommend/{user_id}?k=10      top-k jobs with a "why this job" reason
GET  /features/{user_id}            online features for the user (Feast online store)
GET  /notify/{user_id}              send / don't send, when, and the agent-written message
GET  /metrics                       Prometheus
"""
from __future__ import annotations

import os
import time

import evalwright as ew
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from ..config import load_config

REQS = Counter("pathrank_requests_total", "requests", ["endpoint", "status"])
LAT = Histogram("pathrank_latency_seconds", "latency", ["endpoint"],
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
TOP_SCORE = Histogram("pathrank_top1_score", "ranker score of the top recommendation",
                      buckets=(-4, -2, -1, 0, 1, 2, 4))
SENDS = Counter("pathrank_notifications_total", "notification decisions", ["decision"])

app = FastAPI(title="PathRank", version="0.1.0")
_state: dict = {}

if os.environ.get("EVALWRIGHT_EXPORT"):
    # e.g. EVALWRIGHT_EXPORT=otlp:http://collector:4318/v1/traces or http:<url> or jsonl:<path>
    ew.init("pathrank-api", sample_rate=float(os.environ.get("EVALWRIGHT_SAMPLE_RATE", "1.0")))


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    """One `http` span per request; handler spans (recommend, retrieval, ranking) nest under it."""
    async with ew.span("http", type="http", metadata={"method": request.method, "path": request.url.path}) as s:
        response = await call_next(request)
        route = request.scope.get("route")
        s.name = f"{request.method} {getattr(route, 'path', request.url.path)}"   # template, not raw path
        s.log(metrics={"status_code": response.status_code}, tags=[f"status:{response.status_code // 100}xx"])
        if response.status_code >= 500:
            s.status = "error"
        return response


def svc():
    if "svc" not in _state:
        from ..agent.graph import build_agent
        from ..agent.llm import get_llm
        from .service import Service

        cfg = load_config(os.environ.get("PATHRANK_CONFIG", "configs/careerbuilder.yaml"),
                          os.environ.get("PATHRANK_SET", "").split())   # e.g. "featurestore.online=redis"
        s = Service.from_artifacts(cfg)
        _state["svc"] = s
        _state["agent"] = build_agent(s.profile, lambda u: s.recommend(u, 10)["recommendations"],
                                      lambda u: True, get_llm(cfg.agent.provider, cfg.agent.model),
                                      cfg.agent.max_retries)
    return _state["svc"]


@app.get("/health")
def health():
    s = _state.get("svc")
    return {"status": "ok", "loaded": s is not None, "feature_source": getattr(s, "feature_source", None),
            "towers": getattr(getattr(s, "rec", None), "tower_backend", None)}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    t0 = time.perf_counter()
    try:
        out = svc().recommend(user_id, k)
    except KeyError:
        REQS.labels("recommend", "404").inc()
        raise HTTPException(404, f"unknown user {user_id}")
    if out["recommendations"]:
        TOP_SCORE.observe(out["recommendations"][0]["score"])
    LAT.labels("recommend").observe(time.perf_counter() - t0)
    REQS.labels("recommend", "200").inc()
    return out


@app.get("/features/{user_id}")
def features(user_id: int):
    """Online feature serving: what the ranker sees for this user right now."""
    t0 = time.perf_counter()
    try:
        out = svc().online_features(user_id)
    except KeyError:
        REQS.labels("features", "404").inc()
        raise HTTPException(404, f"unknown user {user_id}")
    LAT.labels("features").observe(time.perf_counter() - t0)
    REQS.labels("features", "200").inc()
    return out


@app.get("/notify/{user_id}")
def notify(user_id: int, threshold: float = 0.3):
    from ..agent.graph import run_agent

    t0 = time.perf_counter()
    try:
        decision = svc().notify_decision(user_id, threshold)
    except KeyError:
        REQS.labels("notify", "404").inc()
        raise HTTPException(404, f"unknown user {user_id}")
    out = {"user_id": user_id, "decision": decision}
    if decision.get("send"):
        r = run_agent(_state["agent"], user_id)
        out["agent_status"] = r.get("status")
        out["message"] = r.get("message") if r.get("status") == "ready" else None
        out["guardrail_errors"] = r.get("errors")
    SENDS.labels("send" if decision.get("send") and out.get("message") else "skip").inc()
    LAT.labels("notify").observe(time.perf_counter() - t0)
    REQS.labels("notify", "200").inc()
    return out


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
