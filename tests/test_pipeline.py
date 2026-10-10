"""End-to-end smoke test on a tiny synthetic dataset: every stage, leakage checks, the API."""
import os

import numpy as np

CFG = "configs/ci.yaml"


def test_results_complete(built):
    from pathrank.utils import load_json
    res = load_json(built.art_dir / "results.json")
    for k in ("retrieval", "rank", "notify", "agent", "monitor"):
        assert k in res
    t = res["rank"]["test"]
    assert t["ranker_v2"]["ndcg@10"] > t["popularity"]["ndcg@10"]
    assert res["agent"]["adversarial_caught"] == "3/3"
    assert res["agent"]["frequency_cap_enforced"]
    assert (built.art_dir / "RESULTS.md").exists() and (built.art_dir / "analytics.html").exists()


def test_asof_has_no_future(built):
    from pathrank.features.pipeline import AsOf, load_store
    s = load_store(built)
    c = s.cutoffs
    ctx = AsOf(s, c["val"], c["test"])
    assert (ctx.hist.ts < c["val"]).all()
    assert ((ctx.labels.ts >= c["val"]) & (ctx.labels.ts < c["test"])).all()
    # popularity features must not count label-window applications
    assert ctx.job_pop_all.sum() == len(ctx.hist)
    # every catalog job is open during the label window
    j = s.jobs.iloc[ctx.catalog]
    assert ((j.start_date < c["test"]) & (j.end_date >= c["val"])).all()


def test_user_history_excludes_labels(built):
    from pathrank.features.pipeline import AsOf, load_store
    s = load_store(built)
    c = s.cutoffs
    ctx = AsOf(s, c["val"], c["test"], built.two_tower.hash_buckets)
    u = ctx.labels.uidx.iloc[0]
    later_titles = set(s.jobs.title.iloc[ctx.labels[ctx.labels.uidx == u].jidx]) - \
        set(s.jobs.title.iloc[ctx.hist[ctx.hist.uidx == u].jidx])
    f = ctx.user_features(np.array([u]))
    from pathrank.utils import hash_tokens
    hist_tok = set(f["tok"][0][f["tok"][0] > 0])
    work = s.users.work_titles.iloc[u] + " | " + " | ".join(s.jobs.title.iloc[ctx.hist[ctx.hist.uidx == u].jidx])
    allowed = set(hash_tokens([work], built.two_tower.hash_buckets, 10_000)[0]) - {0}
    assert hist_tok <= allowed, later_titles


def test_api(built):
    os.environ["PATHRANK_CONFIG"] = CFG
    from fastapi.testclient import TestClient
    from pathrank.serve.app import app
    from pathrank.serve.service import Service
    svc = Service.from_artifacts(built)
    uid = int(svc.store.users.user_id.iloc[svc.eval_users(1, 0)[0]])
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.get(f"/recommend/{uid}?k=5").json()
    assert len(r["recommendations"]) == 5 and all("reason" in x for x in r["recommendations"])
    n = c.get(f"/notify/{uid}?threshold=0.0").json()
    assert "decision" in n
    assert c.get("/recommend/-12345").status_code == 404
    assert b"pathrank_requests_total" in c.get("/metrics").content
