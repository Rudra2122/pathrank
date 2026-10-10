# PathRank results (olx, feature store v1 / ede6063528e6)

Users 300,000 | jobs 137,033 | applications 5,416,311 | text encoder: none (16-d)

Time split: start 2020-02-12, ctx 2020-02-20, val 2020-02-22, test 2020-02-23, end 2020-02-25

Applications per split: retriever_train 896,826, ranker_train 436,961, test 923,890

Data validation: 22 checks, 0 errors, 4 warnings (see data_validation.csv).

## Feature store (Feast) and online feature serving

Online store: sqlite; 300,000 users and 126,214 open jobs materialized at 2020-02-23. Offline snapshots: 304,000 user rows, 384,987 job rows.

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

Online user-feature fetch: p50 0.12 ms, p95 0.13 ms.

## Stage 1: retrieval (test window, each source alone)

| source | recall@100 | ndcg@10 | candidate recall |
|---|---|---|---|
| popularity | 0.0093 | 0.0018 | 0.0093 |
| lp | 0.0056 | 0.0018 | 0.0056 |
| txt | 0.0007 | 0.0001 | 0.0009 |
| tt | 0.0110 | 0.0019 | 0.0248 |
| gcn | 0.0817 | 0.0100 | 0.0817 |
| union_candidates | 0.0347 | 0.0029 | 0.1037 |
Training device: two-tower on cpu (39,045 pairs/s), LightGCN on cpu (2254.7 s).


Avg candidates/user: 543. Test applications to jobs with zero history before the cutoff: 1.2%. LightGCN can score 96.9% of the open catalog and 64.6% of eval users.

(lp = location popularity, txt = text match, tt = two-tower, gcn = LightGCN)

## Stage 2: ranking (test window)

| model | ndcg@10 | map@10 | recall@10 | recall@100 | hit@10 |
|---|---|---|---|---|---|
| popularity | 0.0018 | 0.0008 | 0.0016 | 0.0093 | 0.0115 |
| location_popularity | 0.0018 | 0.0008 | 0.0016 | 0.0056 | 0.0115 |
| two_tower_only | 0.0019 | 0.0008 | 0.0016 | 0.0110 | 0.0127 |
| ranker_v1 | 0.0129 | 0.0057 | 0.0135 | 0.0828 | 0.0732 |
| ranker_v2 | 0.0178 | 0.0081 | 0.0184 | 0.0831 | 0.0950 |

NDCG@10 lift, ranker v2 vs popularity: **880.3%**; vs location popularity: **880.3%**.

Cold users (no history): ndcg@10 0.0003 (n=3172); warm: 0.0259 (n=6828).

### Replayed A/B, ranker v1 (retrieval features) vs v2 (+ context features)

Hash-split arms: diff 0.0056 ndcg@10, 95% CI [0.0030, 0.0080], p=0.000. Paired (same users, both rankers): diff 0.0049, CI [0.0036, 0.0064], relative lift 38.2%.

### SHAP (mean |value|, ranker v2)

| feature | mean abs SHAP |
|---|---|
| job_pop_7d | 0.0918 |
| gcn_score | 0.0676 |
| tt_score | 0.0593 |
| user_days_since_last | 0.0487 |
| txt_rank | 0.0448 |
| title_overlap | 0.0442 |
| user_n_apps | 0.0203 |
| tt_rank | 0.0165 |
| job_age_days | 0.0156 |
| lp_rank | 0.0107 |

Example "why this job" reasons:

- user 1: *job 180419* -> people with similar applications applied here; getting a lot of applications this week
- user 1: *job 82436* -> people with similar applications applied here; getting a lot of applications this week
- user 1: *job 82762* -> people with similar applications applied here; getting a lot of applications this week
- user 86: *job 114741* -> people with similar applications applied here; getting a lot of applications this week
- user 86: *job 58452* -> people with similar applications applied here; matches your profile and past applications
- user 86: *job 166542* -> people with similar applications applied here; getting a lot of applications this week

## Stage 3: fairness re-rank (non-metro job exposure)

Non-metro jobs are 0.0% of the open catalog and 0.0% of test applications.

| min non-metro share per prefix | ndcg@10 | non-metro exposure |
|---|---|---|
| 0.0 | 0.0178 | 0.0% |
| 0.1 | 0.0178 | 0.0% |
| 0.2 | 0.0178 | 0.0% |
| 0.3 | 0.0178 | 0.0% |
| 0.4 | 0.0178 | 0.0% |

User-side: ndcg@10 by degree type

| degree | ndcg@10 | users |
|---|---|---|
| None | 0.0178 | 10000 |

## Inference export (ONNX / Triton / TensorRT)

ONNX vs PyTorch max abs diff: user tower 2.68e-07, item tower 1.64e-07, TensorRT source graph 2.68e-07. Top-10 lists identical with ONNX towers: 99.6%. Triton configs validated against the ModelConfig schema: item_tower, ranker, user_tower, user_tower_trt.

| user tower, CPU | batch 1 p50 (ms) | batch 1024 p50 (ms) |
|---|---|---|
| torch | 0.06 | 0.783 |
| onnxruntime | 1.448 | 3.354 |

GPU numbers (Triton + TensorRT) come from scripts/triton_perf.sh on an NVIDIA GPU.


## Notifications

Apply-within-2-days classifier: ROC-AUC 0.8651 (recency-only baseline 0.8138), PR-AUC 0.7700 (base rate 0.3163).

Days-to-return regression (returners within cap): MAE 0.6481 days, RMSE 0.9242; median baseline MAE 0.7140.

Targeting: reaching 80% of users who apply within 2 days takes notifying 38.5% of users (61.5% fewer sends than notifying everyone). This is targeting, not causal uplift; the data has no randomized sends.

## Notification agent (LangGraph)

Backend: template. 50 users: ready 100.0%, blocked 0.0%, mean attempts 1.0000, judge quality 0.8750 (0-1). Frequency cap enforced: True. Adversarial guardrail checks caught: 3/3.

Run through the evalwright SDK (experiment `20261009-193418408-68ce`, 926 spans traced, case latency p50 43 ms):

| score | mean | 95% CI | n | vs previous run |
|---|---|---|---|---|
| delivered | 1.0000 | [1.000, 1.000] | 50 | +0.000 (0 up / 0 down) |
| first_try | 1.0000 | [1.000, 1.000] | 50 | +0.000 (0 up / 0 down) |
| judge_quality | 0.8750 | [0.875, 0.875] | 50 | +0.000 (0 up / 0 down) |

Gates: all passed.

Template backend: guardrail pass rate is trivially high. Set ANTHROPIC_API_KEY and re-run `--stages agent` for a real number.

## Monitoring (PSI, ranker-train window vs test window)

2 features drifting (PSI > 0.25), 0 to watch. Top: job_age_days 2.3462, user_days_since_last 0.0339, user_n_apps 0.0315, user_cold 0.0278, title_overlap 0.0278

## Model registry and promotion

MLflow registry: ranker v2, two-tower (ONNX) v2; lineage features ede6063528e6, git nogit. Gate: **promoted to champion** (challenger significantly better; paired diff 0.0033, CI [0.001891604904724663, 0.004688328280174231]).

Retrain flag from drift: False.
