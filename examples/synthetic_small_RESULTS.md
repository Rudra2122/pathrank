# PathRank results (synthetic, feature store v1 / a7380f01e8c0)

> **Synthetic data.** The generator plants the structure these models find. Use these numbers to check the pipeline works, never on a resume. Re-run with `configs/careerbuilder.yaml`.

Users 20,000 | jobs 30,000 | applications 163,918 | text encoder: tfidf (64-d)

Time split: start 2012-04-01, ctx 2012-06-05, val 2012-06-20, test 2012-06-25, end 2012-06-30

Applications per split: retriever_train 34,019, ranker_train 14,393, test 18,653

Data validation: 22 checks, 0 errors, 0 warnings (see data_validation.csv).

## Feature store (Feast) and online feature serving

Online store: sqlite; 20,000 users and 8,723 open jobs materialized at 2012-06-25. Offline snapshots: 24,000 user rows, 28,148 job rows.

| parity check (Feast vs feature pipeline) | result |
|---|---|
| pit_parity_tokens_equal | True |
| pit_parity_n_apps_equal | True |
| pit_parity_text_max_abs_diff | 0.0 |
| online_parity_tokens_equal | True |
| online_parity_activity_equal | True |
| online_job_parity_equal | True |
| online_parity_text_max_abs_diff | 0.0 |
| online_parity_num_max_abs_diff | 0.0 |

Online user-feature fetch: p50 0.48 ms, p95 0.58 ms.

## Stage 1: retrieval (test window, each source alone)

| source | recall@100 | ndcg@10 | candidate recall |
|---|---|---|---|
| popularity | 0.0618 | 0.0067 | 0.0618 |
| lp | 0.2051 | 0.0583 | 0.2051 |
| txt | 0.1691 | 0.0157 | 0.2503 |
| tt | 0.4283 | 0.1215 | 0.5895 |
| gcn | 0.0540 | 0.0067 | 0.0540 |
| union_candidates | 0.4007 | 0.0695 | 0.6922 |
Training device: two-tower on cpu (13,836 pairs/s), LightGCN on cpu (29.9 s).


Avg candidates/user: 425. Test applications to jobs with zero history before the cutoff: 18.5%. LightGCN can score 53.3% of the open catalog and 24.7% of eval users.

(lp = location popularity, txt = text match, tt = two-tower, gcn = LightGCN)

## Stage 2: ranking (test window)

| model | ndcg@10 | map@10 | recall@10 | recall@100 | hit@10 |
|---|---|---|---|---|---|
| popularity | 0.0067 | 0.0028 | 0.0069 | 0.0618 | 0.0445 |
| location_popularity | 0.0583 | 0.0287 | 0.0704 | 0.2051 | 0.2695 |
| two_tower_only | 0.1215 | 0.0650 | 0.1408 | 0.4283 | 0.4480 |
| ranker_v1 | 0.1609 | 0.0943 | 0.1597 | 0.4555 | 0.5160 |
| ranker_v2 | 0.2056 | 0.1237 | 0.2063 | 0.5034 | 0.5955 |

NDCG@10 lift, ranker v2 vs popularity: **2978.0%**; vs location popularity: **252.5%**.

Cold users (no history): ndcg@10 0.2031 (n=1114); warm: 0.2088 (n=886).

### Replayed A/B, ranker v1 (retrieval features) vs v2 (+ context features)

Hash-split arms: diff 0.0395 ndcg@10, 95% CI [0.0194, 0.0589], p=0.000. Paired (same users, both rankers): diff 0.0447, CI [0.0385, 0.0510], relative lift 27.8%.

### SHAP (mean |value|, ranker v2)

| feature | mean abs SHAP |
|---|---|
| same_state | 0.3108 |
| tt_rank | 0.2301 |
| job_pop_all | 0.1299 |
| job_pop_7d | 0.1155 |
| tt_score | 0.1096 |
| title_overlap | 0.1095 |
| n_sources | 0.0997 |
| job_days_left | 0.0927 |
| lp_rank | 0.0728 |
| user_degree | 0.0681 |

Example "why this job" reasons:

- user 22: *Junior Recruiter Talent* -> it's in your state; matches your profile and past applications
- user 22: *Junior Talent Resources* -> popular near you; it's in your state
- user 22: *Lead Human Recruiter* -> it's in your state; matches your profile and past applications
- user 29: *Senior Sales Inside* -> it's in your state; matches your profile and past applications
- user 29: *Lead Representative Executive* -> it's in your state; matches your profile and past applications
- user 29: *Junior Inside Sales* -> it's in your state; matches your profile and past applications

## Stage 3: fairness re-rank (non-metro job exposure)

Non-metro jobs are 51.1% of the open catalog and 44.4% of test applications.

| min non-metro share per prefix | ndcg@10 | non-metro exposure |
|---|---|---|
| 0.0 | 0.2056 | 40.6% |
| 0.1 | 0.2048 | 41.8% |
| 0.2 | 0.2032 | 43.7% |
| 0.3 | 0.2020 | 45.7% |
| 0.4 | 0.2007 | 48.4% |

User-side: ndcg@10 by degree type

| degree | ndcg@10 | users |
|---|---|---|
| Associate's | 0.2165 | 301 |
| Bachelor's | 0.2041 | 464 |
| High School | 0.1964 | 581 |
| Master's | 0.2005 | 156 |
| None | 0.2142 | 358 |
| PhD | 0.2038 | 24 |
| Vocational | 0.2103 | 116 |

## Inference export (ONNX / Triton / TensorRT)

ONNX vs PyTorch max abs diff: user tower 1.19e-07, item tower 5.96e-08, TensorRT source graph 1.19e-07. Top-10 lists identical with ONNX towers: 100.0%. Triton configs validated against the ModelConfig schema: item_tower, ranker, user_tower, user_tower_trt.

| user tower, CPU | batch 1 p50 (ms) | batch 1024 p50 (ms) |
|---|---|---|
| torch | 0.266 | 3.267 |
| onnxruntime | 0.041 | 3.279 |

GPU numbers (Triton + TensorRT) come from scripts/triton_perf.sh on an NVIDIA GPU.


## Notifications

Apply-within-7-days classifier: ROC-AUC 0.9389 (recency-only baseline 0.9284), PR-AUC 0.6108 (base rate 0.0656).

Days-to-return regression (returners within cap): MAE 2.0715 days, RMSE 3.5176; median baseline MAE 2.9984.

Targeting: reaching 80% of next-week applicants takes notifying 9.2% of users (90.8% fewer sends than notifying everyone). This is targeting, not causal uplift; the data has no randomized sends.

## Notification agent (LangGraph)

Backend: template. 20 users: ready 100.0%, blocked 0.0%, mean attempts 1.0000, judge quality 0.8750 (0-1). Frequency cap enforced: True. Adversarial guardrail checks caught: 3/3.

Run through the evalwright SDK (experiment `20261009-041043951-000e`, 416 spans traced, case latency p50 440 ms):

| score | mean | 95% CI | n | vs previous run |
|---|---|---|---|---|
| delivered | 1.0000 | [1.000, 1.000] | 20 | - |
| first_try | 1.0000 | [1.000, 1.000] | 20 | - |
| judge_quality | 0.8750 | [0.875, 0.875] | 20 | - |

Gates: all passed.

Template backend: guardrail pass rate is trivially high. Set ANTHROPIC_API_KEY and re-run `--stages agent` for a real number.

## Monitoring (PSI, ranker-train window vs test window)

0 features drifting (PSI > 0.25), 1 to watch. Top: user_days_since_last 0.1731, gcn_score 0.0736, user_cold 0.0183, user_n_apps 0.0163, job_age_days 0.0142

## Model registry and promotion

MLflow registry: ranker v1, two-tower (ONNX) v1; lineage features a7380f01e8c0, git nogit. Gate: **promoted to champion** (first registered model).

Retrain flag from drift: False.
