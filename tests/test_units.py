import numpy as np
import pandas as pd

from pathrank.agent.guardrails import check
from pathrank.eval.metrics import per_user_metrics
from pathrank.notify.policy import FrequencyCap, coverage_curve, sponsored_slot
from pathrank.rerank.fairness import rerank

RECS = [{"job_id": 1, "title": "Registered Nurse", "city": "austin", "state": "TX", "reason": "x"},
        {"job_id": 2, "title": "Data Analyst", "city": "austin", "state": "TX", "reason": "y"}]


def test_ndcg_perfect_and_miss():
    ranked = pd.DataFrame({"uidx": [0, 0, 1, 1], "jidx": [10, 11, 20, 21], "score": [2, 1, 2, 1]})
    pu = per_user_metrics(ranked, {0: {10}, 1: {99}}, np.array([0, 1])).set_index("uidx")
    assert pu.loc[0, "ndcg@10"] == 1.0
    assert pu.loc[1, "ndcg@10"] == 0.0       # positive never retrieved still counts
    assert pu.loc[0, "recall@100"] == 1.0


def test_ndcg_second_position():
    ranked = pd.DataFrame({"uidx": [0, 0], "jidx": [10, 11], "score": [2, 1]})
    pu = per_user_metrics(ranked, {0: {11}}, np.array([0]))
    assert abs(pu["ndcg@10"].iloc[0] - 1 / np.log2(3)) < 1e-9


def test_guardrails():
    ok = {"subject": 'New: "Registered Nurse"', "body": 'See "Data Analyst" and apply.', "job_ids": [1, 2]}
    assert check(ok, RECS) == []
    assert any("unrecommended_title" in e for e in check({**ok, "body": 'Try "Astronaut"'}, RECS))
    assert "mentions_unrecommended_job_id" in check({**ok, "job_ids": [7]}, RECS)
    assert "states_pay_not_in_data" in check({**ok, "body": "pays $50k"}, RECS)
    assert "contains_contact_info" in check({**ok, "body": "call 512-555-0100"}, RECS)
    assert check({"body": "x"}, RECS) == ["malformed_message"]


def test_fairness_prefix_constraint():
    ranked = pd.DataFrame({"uidx": [0] * 20, "jidx": np.arange(20), "score": -np.arange(20.0)})
    prot = np.zeros(20, bool)
    prot[15:] = True                           # protected jobs ranked last
    out = rerank(ranked, prot, 0.4, k=10, pool=20)
    for p in range(1, 11):
        assert prot[out.jidx.to_numpy()[:p]].sum() >= int(np.floor(0.4 * p))
    base = rerank(ranked, prot, 0.0, k=10, pool=20)
    assert list(base.jidx) == list(range(10))  # no constraint -> original order


def test_frequency_cap():
    cap = FrequencyCap(2)
    for t in (0, 1):
        assert cap.allowed(7, now=t)
        cap.record(7, now=t)
    assert not cap.allowed(7, now=2)
    assert cap.allowed(7, now=8 * 86400)      # window slides


def test_coverage_and_sponsored():
    df = pd.DataFrame({"p_apply": [0.9, 0.8, 0.1, 0.05], "y_apply": [1, 1, 0, 0]})
    c = coverage_curve(df, targets=(0.5, 1.0))
    assert c["send_frac_for_100pct_applicants"] == 0.5
    s = sponsored_slot(np.array([0.3, 0.2, 0.1]), 0.01, bid=0.01)
    assert s["insert"] is False


def test_frequency_cap_is_atomic_under_threads():
    from concurrent.futures import ThreadPoolExecutor
    cap = FrequencyCap(2)
    with ThreadPoolExecutor(16) as pool:
        granted = sum(pool.map(lambda _: cap.try_acquire(42, now=100.0), range(200)))
    assert granted == 2


def test_guardrails_allow_numeric_job_titles():
    """XING/OLX jobs are titled "job <id>"; a 10-digit id must not read as a phone number."""
    from pathrank.agent.guardrails import check
    recs = [{"job_id": 1716613042, "title": "job 1716613042", "city": "", "state": "", "reason": "x"}]
    msg = {"subject": 'New match: "job 1716613042"', "body": 'See "job 1716613042".', "job_ids": [1716613042]}
    assert check(msg, recs) == []
    assert "contains_contact_info" in check({**msg, "body": "call 512-555-0100"}, recs)


def test_inferred_job_window_ignores_the_future():
    """XING/OLX: catalog and job_left at cutoff t must not change when post-t interactions change."""
    from types import SimpleNamespace

    import pandas as pd

    from pathrank.features.pipeline import job_window
    T = pd.Timestamp("2020-02-10")
    jobs = pd.DataFrame({"start_date": pd.to_datetime(["2020-02-01", "2020-02-01", "2020-02-10 12:00:00"], format="mixed"),
                         "end_date": pd.to_datetime(["2020-02-20", "2020-03-01", "2020-03-01"])})
    past = pd.DataFrame({"jidx": [0, 1], "ts": pd.to_datetime(["2020-02-08", "2020-02-01"])})
    future = pd.DataFrame({"jidx": [1, 1, 2], "ts": pd.to_datetime(["2020-02-11", "2020-02-12", "2020-02-11"])})
    mk = lambda apps: SimpleNamespace(jobs=jobs, apps=apps, manifest={"source": "olx"})  # noqa: E731
    a = job_window(mk(past), T, T + pd.Timedelta(days=2))
    b = job_window(mk(pd.concat([past, future])), T, T + pd.Timedelta(days=2))
    assert list(a[0]) == list(b[0]) == [0, 2]     # job 1 inactive for 9 days -> out; job 2 newly posted -> in
    assert np.isnan(a[2]).all() and np.isnan(b[2]).all()
