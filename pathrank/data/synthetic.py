"""Generate a CareerBuilder-schema dataset at any scale (tested to 10M+ applications).

Used for (1) CI / tests without Kaggle credentials, (2) throughput benchmarks beyond the
real dataset's size. Numbers measured on this data are NOT resume numbers: the generator
plants the structure the models then find.

Planted structure (so every model has something real to learn):
  * users prefer 1-2 job categories (linked to major + past titles)
  * strong locality: same city > same state > anywhere
  * jobs have a latent quality; postings live 14-45 days; users apply in sessions
  * degree type shifts locality and category mix (gives the fairness analysis something to find)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import log

CATEGORIES = {
    "nursing": ("registered nurse rn lpn patient care clinical icu", "patient care medication charting hospital clinical shifts licensure bls"),
    "software": ("software engineer developer java python web full stack", "code apis databases agile cloud testing deploy services git"),
    "sales": ("sales representative account executive inside outside retail", "quota pipeline clients prospecting crm territory commission revenue"),
    "accounting": ("accountant staff senior accounts payable receivable bookkeeper", "ledger reconciliation gaap audit invoices tax reporting excel"),
    "customer_service": ("customer service representative call center support agent", "phones inquiries resolve complaints tickets chat empathy schedule"),
    "warehouse": ("warehouse associate forklift operator picker packer loader", "inventory shipping receiving pallets lifting safety scanners shifts"),
    "trucking": ("truck driver cdl class a otr regional delivery", "routes miles hauling logbook dot safety freight home time"),
    "admin": ("administrative assistant receptionist office clerk secretary", "filing scheduling phones calendars data entry microsoft office"),
    "teaching": ("teacher tutor instructor substitute teaching assistant", "classroom curriculum students lesson plans grading certification"),
    "marketing": ("marketing coordinator manager digital social media specialist", "campaigns content seo brand analytics email advertising"),
    "finance": ("financial analyst advisor banker loan officer underwriter", "models forecasting portfolios credit risk lending investments"),
    "hr": ("human resources generalist recruiter coordinator talent", "hiring onboarding benefits payroll employee relations compliance"),
    "mechanic": ("automotive technician mechanic diesel service tech", "repair engines diagnostics brakes tools ase vehicles maintenance"),
    "retail": ("retail store associate cashier assistant manager merchandiser", "sales floor register inventory displays customers weekends"),
    "food": ("cook chef server line prep restaurant crew", "kitchen menu food safety prep orders dining shifts"),
    "security": ("security officer guard patrol loss prevention", "patrol surveillance reports access control safety licensed"),
    "medical_admin": ("medical assistant billing coder receptionist", "insurance claims coding icd records scheduling patients ehr"),
    "engineering": ("mechanical electrical civil engineer design", "cad drawings specifications projects manufacturing testing"),
    "construction": ("construction laborer carpenter electrician plumber foreman", "sites tools blueprints framing wiring safety osha"),
    "data": ("data analyst scientist business intelligence sql", "sql dashboards statistics python reporting tableau insights"),
}
GENERIC = ("team work environment opportunity growth benefits full time part time experience "
           "skills communication apply today competitive pay company position role responsibilities").split()
SENIORITY = ["", "", "", "senior", "junior", "lead", "entry level", "ii"]
DEGREES = ["None", "High School", "Vocational", "Associate's", "Bachelor's", "Master's", "PhD"]
DEG_P = np.array([0.18, 0.30, 0.06, 0.14, 0.24, 0.07, 0.01])
# degree-index-specific category preference tilt (index into CATS)
CATS = list(CATEGORIES)
HIGH_ED = {"software", "accounting", "teaching", "marketing", "finance", "hr", "engineering", "data", "nursing"}

BASE = pd.Timestamp("2012-04-01")
DAYS = 91
STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA",
          "TN", "IN", "MO", "MD", "WI", "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "IA",
          "UT", "NV", "AR", "MS", "KS", "NM", "NE", "WV", "ID", "HI", "NH", "ME", "RI", "MT", "DE",
          "SD", "ND", "AK", "VT", "WY"]


def _zipf(n, a, rng):
    w = 1.0 / np.arange(1, n + 1) ** a
    w = w / w.sum()
    return rng.permutation(w) if False else w


def generate(out_dir: str, n_users: int = 50_000, n_jobs: int = 80_000, n_apps: int = 500_000,
             seed: int = 0) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_cities = max(200, min(3000, n_jobs // 30))
    C = len(CATS)

    # ---- geography -------------------------------------------------------
    city_state = rng.choice(len(STATES), n_cities, p=_zipf(len(STATES), 1.0, rng))
    city_w = _zipf(n_cities, 1.1, rng)
    city_names = np.array([f"city{c:04d}" for c in range(n_cities)])
    city_zip3 = np.array([f"{(100 + c) % 1000:03d}" for c in range(n_cities)])

    # ---- jobs -----------------------------------------------------------
    job_cat = rng.choice(C, n_jobs, p=_zipf(C, 0.6, rng))
    job_city = rng.choice(n_cities, n_jobs, p=city_w)
    job_start = rng.integers(-30, DAYS, n_jobs)
    job_dur = rng.integers(14, 46, n_jobs)
    job_end = job_start + job_dur
    job_q = rng.lognormal(0, 0.8, n_jobs)
    job_ids = rng.permutation(np.arange(1, n_jobs * 2))[:n_jobs]

    title_words = [CATEGORIES[c][0].split() for c in CATS]
    desc_words = [CATEGORIES[c][1].split() for c in CATS]
    titles, descs = [], []
    for c in job_cat:
        tw = title_words[c]
        t = " ".join(filter(None, [SENIORITY[rng.integers(len(SENIORITY))],
                                   *rng.choice(tw, 2, replace=False)])).title()
        titles.append(t)
        n_cat = rng.integers(8, 16)
        words = list(rng.choice(desc_words[c], n_cat)) + list(rng.choice(GENERIC, rng.integers(6, 14)))
        rng.shuffle(words)
        descs.append(" ".join(words).capitalize() + ".")
    jobs = pd.DataFrame({
        "JobID": job_ids, "WindowID": np.clip((job_start.clip(0) // 13) + 1, 1, 7),
        "Title": titles, "Description": descs,
        "Requirements": [f"{rng.integers(0, 6)}+ years experience" for _ in range(n_jobs)],
        "City": city_names[job_city], "State": np.array(STATES)[city_state[job_city]],
        "Country": "US", "Zip5": [z + "01" for z in city_zip3[job_city]],
        "StartDate": (BASE + pd.to_timedelta(job_start, "D")).strftime("%Y-%m-%d %H:%M:%S.000"),
        "EndDate": (BASE + pd.to_timedelta(job_end, "D")).strftime("%Y-%m-%d %H:%M:%S.000"),
    })

    # ---- users ----------------------------------------------------------
    deg = rng.choice(len(DEGREES), n_users, p=DEG_P)
    high = deg >= 4
    cat_p_high = np.array([3.0 if c in HIGH_ED else 0.6 for c in CATS]); cat_p_high /= cat_p_high.sum()
    cat_p_low = np.array([0.5 if c in HIGH_ED else 2.0 for c in CATS]); cat_p_low /= cat_p_low.sum()
    pref1 = np.where(high, rng.choice(C, n_users, p=cat_p_high), rng.choice(C, n_users, p=cat_p_low))
    pref2 = np.where(rng.random(n_users) < 0.5, rng.choice(C, n_users), pref1)
    u_city = rng.choice(n_cities, n_users, p=city_w)
    # non-degree users are more local
    p_city = np.where(high, 0.40, 0.65)
    p_state = np.where(high, 0.30, 0.25)
    activity = rng.lognormal(0, 1.1, n_users)
    first_day = rng.integers(0, DAYS - 5, n_users)
    span = np.minimum(DAYS - first_day, rng.integers(3, 60, n_users))
    user_ids = rng.permutation(np.arange(1, n_users * 2))[:n_users]
    years = np.clip(rng.gamma(2.0, 3.0, n_users) - (deg == 0) * 1.0, 0, 45).round()
    majors = np.where(high, np.array([CATS[c].replace("_", " ") for c in pref1]), "Not Applicable")
    users = pd.DataFrame({
        "UserID": user_ids, "WindowID": np.clip(first_day // 13 + 1, 1, 7), "Split": "Train",
        "City": city_names[u_city], "State": np.array(STATES)[city_state[u_city]], "Country": "US",
        "ZipCode": [z + "01" for z in city_zip3[u_city]], "DegreeType": np.array(DEGREES)[deg],
        "Major": majors, "GraduationDate": "",
        "WorkHistoryCount": rng.integers(0, 8, n_users), "TotalYearsExperience": years,
        "CurrentlyEmployed": np.where(rng.random(n_users) < 0.45, "Yes", "No"),
        "ManagedOthers": np.where(rng.random(n_users) < 0.2, "Yes", "No"), "ManagedHowMany": 0,
    })
    # work history: 0-4 past titles in preferred categories
    n_hist = rng.integers(0, 5, n_users)
    hu = np.repeat(np.arange(n_users), n_hist)
    hc = np.where(rng.random(len(hu)) < 0.75, pref1[hu], rng.choice(C, len(hu)))
    hist = pd.DataFrame({
        "UserID": user_ids[hu], "WindowID": 1, "Split": "Train",
        "Sequence": np.concatenate([np.arange(k) + 1 for k in n_hist]) if len(hu) else [],
        "JobTitle": [" ".join(rng.choice(title_words[c], 2, replace=False)).title() for c in hc],
    })

    # ---- applications -----------------------------------------------------
    w = activity / activity.sum()
    app_u = rng.choice(n_users, n_apps, p=w)
    n_sess = np.maximum(1, np.ceil(np.bincount(app_u, minlength=n_users) / 3)).astype(int)
    sess_idx = (rng.random(n_apps) * n_sess[app_u]).astype(int)
    frac = ((user_ids[app_u] * 7919 + sess_idx * 104729) % 997) / 997.0
    app_day = first_day[app_u] + np.floor(frac * span[app_u]).astype(int)
    app_cat = np.where(rng.random(n_apps) < 0.85,
                       np.where(rng.random(n_apps) < 0.65, pref1[app_u], pref2[app_u]),
                       rng.choice(C, n_apps))
    r = rng.random(n_apps)
    scope = np.where(r < p_city[app_u], 0, np.where(r < p_city[app_u] + p_state[app_u], 1, 2))

    # bucket keys: scope0 = (cat, city), scope1 = (cat, state), scope2 = (cat)
    jstate = city_state[job_city]
    keys_by_scope = [job_cat * n_cities + job_city, job_cat * len(STATES) + jstate, job_cat]
    u_state = city_state[u_city]
    akey_by_scope = [app_cat * n_cities + u_city[app_u], app_cat * len(STATES) + u_state[app_u], app_cat]
    sorted_by_scope = []
    for k in keys_by_scope:
        comp = k.astype(np.int64) * 1000 + (job_start + 50)
        order = np.argsort(comp, kind="stable")
        sorted_by_scope.append((comp[order], order))

    chosen = np.full(n_apps, -1, dtype=np.int64)
    qmax = np.quantile(job_q, 0.99)
    pending = np.arange(n_apps)
    for it in range(40):
        if len(pending) == 0:
            break
        sc = np.minimum(scope[pending] + it // 8, 2)
        for s in (0, 1, 2):
            m = pending[sc == s]
            if len(m) == 0:
                continue
            comp, order = sorted_by_scope[s]
            base = akey_by_scope[s][m].astype(np.int64) * 1000
            lo = np.searchsorted(comp, base + (app_day[m] - 46 + 50), "left")
            hi = np.searchsorted(comp, base + (app_day[m] + 50), "right")
            ok = hi > lo
            pick = lo + (rng.random(len(m)) * np.maximum(hi - lo, 1)).astype(np.int64)
            pick = np.minimum(pick, len(order) - 1)
            j = order[pick]
            ok &= job_end[j] >= app_day[m]
            ok &= rng.random(len(m)) < np.minimum(job_q[j] / qmax, 1.0)
            chosen[m[ok]] = j[ok]
        pending = np.where(chosen < 0)[0]
    keep = chosen >= 0
    log.info("synthetic: placed %s / %s applications", f"{keep.sum():,}", f"{n_apps:,}")
    secs = rng.integers(0, 86400, n_apps)
    ts = BASE + pd.to_timedelta(app_day * 86400 + secs, "s")
    apps = pd.DataFrame({
        "UserID": user_ids[app_u[keep]], "WindowID": np.clip(app_day[keep] // 13 + 1, 1, 7),
        "Split": "Train", "ApplicationDate": ts[keep].strftime("%Y-%m-%d %H:%M:%S.000"),
        "JobID": job_ids[chosen[keep]],
    }).drop_duplicates(["UserID", "JobID"])

    for name, df in [("apps.tsv", apps), ("users.tsv", users), ("user_history.tsv", hist), ("jobs.tsv", jobs)]:
        df.to_csv(out / name, sep="\t", index=False)
    stats = {"users": len(users), "jobs": len(jobs), "apps": len(apps), "user_history": len(hist)}
    log.info("synthetic dataset written to %s: %s", out, stats)
    return stats
