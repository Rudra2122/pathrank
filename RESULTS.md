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

Online user-feature fetch: p50 0.12 ms, p95 0.14 ms.

## Stage 1: retrieval (test window, each source alone)

| source | recall@100 | ndcg@10 | candidate recall |
|---|---|---|---|
| popularity | 0.0093 | 0.0018 | 0.0093 |
| lp | 0.0056 | 0.0018 | 0.0056 |
| txt | 0.0007 | 0.0001 | 0.0009 |
| tt | 0.0110 | 0.0019 | 0.0248 |
| gcn | 0.0824 | 0.0094 | 0.0824 |
| union_candidates | 0.0358 | 0.0030 | 0.1045 |
Training device: two-tower on cpu (37,008 pairs/s), LightGCN on cpu (4856.4 s).


Avg candidates/user: 543. Test applications to jobs with zero history before the cutoff: 1.2%. LightGCN can score 96.9% of the open catalog and 64.6% of eval users.

(lp = location popularity, txt = text match, tt = two-tower, gcn = LightGCN)

## Stage 2: ranking (test window)

| model | ndcg@10 | map@10 | recall@10 | recall@100 | hit@10 |
|---|---|---|---|---|---|
| popularity | 0.0018 | 0.0008 | 0.0016 | 0.0093 | 0.0115 |
| location_popularity | 0.0018 | 0.0008 | 0.0016 | 0.0056 | 0.0115 |
| two_tower_only | 0.0019 | 0.0008 | 0.0016 | 0.0110 | 0.0127 |
| ranker_v1 | 0.0126 | 0.0055 | 0.0132 | 0.0841 | 0.0729 |
| ranker_v2 | 0.0188 | 0.0083 | 0.0199 | 0.0853 | 0.1021 |

NDCG@10 lift, ranker v2 vs popularity: **934.2%**; vs location popularity: **934.2%**.

Cold users (no history): ndcg@10 0.0002 (n=3172); warm: 0.0274 (n=6828).

### Replayed A/B, ranker v1 (retrieval features) vs v2 (+ context features)

Hash-split arms: diff 0.0070 ndcg@10, 95% CI [0.0046, 0.0094], p=0.000. Paired (same users, both rankers): diff 0.0061, CI [0.0049, 0.0074], relative lift 48.6%.

### SHAP (mean |value|, ranker v2)

| feature | mean abs SHAP |
|---|---|
| title_overlap | 0.5891 |
| job_pop_7d | 0.3433 |
| gcn_score | 0.3207 |
| tt_score | 0.3033 |
| user_days_since_last | 0.1892 |
| user_n_apps | 0.1504 |
| txt_rank | 0.1191 |
| tt_rank | 0.0871 |
| job_age_days | 0.0655 |
| job_pop_all | 0.0590 |

Example "why this job" reasons:

- user 1: *job 169497* -> people with similar applications applied here; getting a lot of applications this week
- user 1: *job 82762* -> people with similar applications applied here; getting a lot of applications this week
- user 1: *job 82436* -> people with similar applications applied here; getting a lot of applications this week
- user 86: *job 44589* -> people with similar applications applied here; getting a lot of applications this week
- user 86: *job 146397* -> people with similar applications applied here; getting a lot of applications this week
- user 86: *job 176931* -> people with similar applications applied here; getting a lot of applications this week

## Stage 3: fairness re-rank (non-metro job exposure)

Non-metro jobs are 0.0% of the open catalog and 0.0% of test applications.

| min non-metro share per prefix | ndcg@10 | non-metro exposure |
|---|---|---|
| 0.0 | 0.0188 | 0.0% |
| 0.1 | 0.0188 | 0.0% |
| 0.2 | 0.0188 | 0.0% |
| 0.3 | 0.0188 | 0.0% |
| 0.4 | 0.0188 | 0.0% |

User-side: ndcg@10 by degree type

| degree | ndcg@10 | users |
|---|---|---|
| None | 0.0188 | 10000 |

## Inference export (ONNX / Triton / TensorRT)

ONNX vs PyTorch max abs diff: user tower 2.68e-07, item tower 1.64e-07, TensorRT source graph 2.68e-07. Top-10 lists identical with ONNX towers: 98.2%. Triton configs validated against the ModelConfig schema: item_tower, ranker, user_tower, user_tower_trt.

| user tower, CPU | batch 1 p50 (ms) | batch 1024 p50 (ms) |
|---|---|---|
| torch | 0.06 | 0.741 |
| onnxruntime | 1.486 | 3.321 |

GPU numbers (Triton + TensorRT) come from scripts/triton_perf.sh on an NVIDIA GPU.


## Notifications

Apply-within-2-days classifier: ROC-AUC 0.8651 (recency-only baseline 0.8138), PR-AUC 0.7700 (base rate 0.3163).

Days-to-return regression (returners within cap): MAE 0.6481 days, RMSE 0.9242; median baseline MAE 0.7140.

Targeting: reaching 80% of users who apply within 2 days takes notifying 38.5% of users (61.5% fewer sends than notifying everyone). This is targeting, not causal uplift; the data has no randomized sends.

## Notification agent (LangGraph)

Backend: openai. 50 users: ready 100.0%, blocked 0.0%, mean attempts 1.0000, judge quality 0.8575 (0-1). Frequency cap enforced: True. Adversarial guardrail checks caught: 3/3.

Run through the evalwright SDK (experiment `20261010-023929131-259a`, 926 spans traced, case latency p50 2983 ms):

| score | mean | 95% CI | n | vs previous run |
|---|---|---|---|---|
| delivered | 1.0000 | [1.000, 1.000] | 50 | +0.000 (0 up / 0 down) |
| first_try | 1.0000 | [1.000, 1.000] | 50 | +0.000 (0 up / 0 down) |
| judge_quality | 0.8575 | [0.825, 0.887] | 50 | -0.018 (32 up / 18 down) |

Gates: all passed.

## TensorFlow baseline (Keras two-tower, same features and loss as the PyTorch tower)

| model | recall@100 | ndcg@10 | train time (s) |
|---|---|---|---|
| PyTorch two-tower | 0.0110 | 0.0019 | 74.17 |
| TensorFlow/Keras two-tower | 0.0161 | 0.0032 | 87.26 |

TF device: CPU, 896,826 training pairs, 3 epochs, eval on the same 10,000 test users.

## Monitoring (PSI, ranker-train window vs test window)

2 features drifting (PSI > 0.25), 0 to watch. Top: job_age_days 2.3348, user_days_since_last 0.0362, user_n_apps 0.0323, tt_score 0.0294, user_cold 0.0274

Evidently 0.7.23 data-drift report (`drift_report.html`): 5 of 21 ranker features flagged (23.8%). Top by Evidently's per-column test: job_age_days 0.6034 (Wasserstein distance (normed)), user_days_since_last 0.1945 (Wasserstein distance (normed)), tt_score 0.1864 (Wasserstein distance (normed)), user_n_apps 0.167 (Wasserstein distance (normed)), job_pop_all 0.1046 (Wasserstein distance (normed)). Skipped (always empty): job_days_left

## Model registry and promotion

MLflow registry: ranker v3, two-tower (ONNX) v3; lineage features ede6063528e6, git 59eba6e. Gate: **kept existing champion** (CI lower bound -0.0008 not above margin 0.0; paired diff 0.0003, CI [-0.0008149384453024134, 0.0014565425675034395]).

Retrain flag from drift: False.
