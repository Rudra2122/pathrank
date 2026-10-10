# PathRank results (olx, feature store v1 / 2069887abe7e)

Users 300,000 | jobs 137,290 | applications 5,412,323 | text encoder: none (16-d)

Time split: start 2020-02-12, ctx 2020-02-20, val 2020-02-22, test 2020-02-23, end 2020-02-25

Applications per split: retriever_train 897,452, ranker_train 436,254, test 925,130

Data validation: 22 checks, 0 errors, 4 warnings (see data_validation.csv).

## Feature store (Feast) and online feature serving

Online store: sqlite; 300,000 users and 126,460 open jobs materialized at 2020-02-23. Offline snapshots: 304,000 user rows, 385,969 job rows.

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

Online user-feature fetch: p50 1.01 ms, p95 1.06 ms.

## Stage 1: retrieval (test window, each source alone)

| source | recall@100 | ndcg@10 | candidate recall |
|---|---|---|---|
| popularity | 0.0100 | 0.0019 | 0.0100 |
| lp | 0.0056 | 0.0019 | 0.0056 |
| txt | 0.0006 | 0.0001 | 0.0010 |
| tt | 0.0104 | 0.0019 | 0.0240 |
| gcn | 0.0795 | 0.0082 | 0.1424 |
| union_candidates | 0.0348 | 0.0023 | 0.1611 |
Training device: two-tower on cuda:NVIDIA A100-SXM4-40GB (191,973 pairs/s), LightGCN on cuda:NVIDIA A100-SXM4-40GB (75.4 s).


Avg candidates/user: 669. Test applications to jobs with zero history before the cutoff: 1.2%. LightGCN can score 96.9% of the open catalog and 64.2% of eval users.

(lp = location popularity, txt = text match, tt = two-tower, gcn = LightGCN)

## Stage 2: ranking (test window)

| model | ndcg@10 | map@10 | recall@10 | recall@100 | hit@10 |
|---|---|---|---|---|---|
| popularity | 0.0019 | 0.0008 | 0.0016 | 0.0100 | 0.0135 |
| location_popularity | 0.0019 | 0.0008 | 0.0016 | 0.0056 | 0.0135 |
| two_tower_only | 0.0019 | 0.0007 | 0.0019 | 0.0104 | 0.0132 |
| ranker_v1 | 0.0128 | 0.0056 | 0.0124 | 0.0710 | 0.0746 |
| ranker_v2 | 0.0179 | 0.0077 | 0.0195 | 0.0977 | 0.1007 |

NDCG@10 lift, ranker v2 vs popularity: **830.1%**; vs location popularity: **830.1%**.

Cold users (no history): ndcg@10 0.0005 (n=3175); warm: 0.0259 (n=6825).

### Replayed A/B, ranker v1 (retrieval features) vs v2 (+ context features)

Hash-split arms: diff 0.0051 ndcg@10, 95% CI [0.0027, 0.0075], p=0.000. Paired (same users, both rankers): diff 0.0051, CI [0.0039, 0.0064], relative lift 40.2%.

### SHAP (mean |value|, ranker v2)

| feature | mean abs SHAP |
|---|---|
| gcn_score | 0.3164 |
| job_pop_7d | 0.2009 |
| user_days_since_last | 0.1844 |
| tt_score | 0.0811 |
| txt_rank | 0.0663 |
| user_n_apps | 0.0642 |
| title_overlap | 0.0482 |
| gcn_rank | 0.0404 |
| tt_rank | 0.0289 |
| job_age_days | 0.0211 |

Example "why this job" reasons:

- user 3: *job 156594* -> people with similar applications applied here; getting a lot of applications this week
- user 3: *job 35418* -> people with similar applications applied here; getting a lot of applications this week
- user 3: *job 93074* -> people with similar applications applied here; getting a lot of applications this week
- user 127: *job 103061* -> people with similar applications applied here; getting a lot of applications this week
- user 127: *job 47751* -> people with similar applications applied here; getting a lot of applications this week
- user 127: *job 86949* -> people with similar applications applied here; getting a lot of applications this week

## Stage 3: fairness re-rank (non-metro job exposure)

Non-metro jobs are 0.0% of the open catalog and 0.0% of test applications.

| min non-metro share per prefix | ndcg@10 | non-metro exposure |
|---|---|---|
| 0.0 | 0.0179 | 0.0% |
| 0.1 | 0.0179 | 0.0% |
| 0.2 | 0.0179 | 0.0% |
| 0.3 | 0.0179 | 0.0% |
| 0.4 | 0.0179 | 0.0% |

User-side: ndcg@10 by degree type

| degree | ndcg@10 | users |
|---|---|---|
| None | 0.0179 | 10000 |

## Inference export (ONNX / Triton / TensorRT)

ONNX vs PyTorch max abs diff: user tower 8.94e-08, item tower 5.96e-08, TensorRT source graph 8.94e-08. Top-10 lists identical with ONNX towers: 100.0%. Triton configs validated against the ModelConfig schema: item_tower, ranker, user_tower, user_tower_trt.

| user tower, CPU | batch 1 p50 (ms) | batch 1024 p50 (ms) |
|---|---|---|
| torch | 0.449 | 1.741 |
| onnxruntime | 0.07 | 2.415 |

GPU numbers (Triton + TensorRT) come from scripts/triton_perf.sh on an NVIDIA GPU.


## Notifications

Apply-within-2-days classifier: ROC-AUC 0.8643 (recency-only baseline 0.8146), PR-AUC 0.7703 (base rate 0.3168).

Days-to-return regression (returners within cap): MAE 0.6449 days, RMSE 0.9133; median baseline MAE 0.7081.

Targeting: reaching 80% of users who apply within 2 days takes notifying 40.1% of users (59.9% fewer sends than notifying everyone). This is targeting, not causal uplift; the data has no randomized sends.

## Notification agent (LangGraph)

Backend: template. 50 users: ready 100.0%, blocked 0.0%, mean attempts 1.0000, judge quality 0.8750 (0-1). Frequency cap enforced: True. Adversarial guardrail checks caught: 3/3.

Run through the evalwright SDK (experiment `20261010-043529517-99e4`, 926 spans traced, case latency p50 323 ms):

| score | mean | 95% CI | n | vs previous run |
|---|---|---|---|---|
| delivered | 1.0000 | [1.000, 1.000] | 50 | +0.000 (0 up / 0 down) |
| first_try | 1.0000 | [1.000, 1.000] | 50 | +0.000 (0 up / 0 down) |
| judge_quality | 0.8750 | [0.875, 0.875] | 50 | +0.000 (0 up / 0 down) |

Gates: all passed.

Template backend: guardrail pass rate is trivially high. Set ANTHROPIC_API_KEY and re-run `--stages agent` for a real number.

## Monitoring (PSI, ranker-train window vs test window)

2 features drifting (PSI > 0.25), 0 to watch. Top: job_age_days 2.3847, user_n_apps 0.0474, tt_score 0.0369, title_overlap 0.0366, user_cold 0.0366

## Model registry and promotion

MLflow registry: ranker v5, two-tower (ONNX) v5; lineage features 2069887abe7e, git nogit. Gate: **kept existing champion** (CI lower bound -0.0032 not above margin 0.0; paired diff -0.0020, CI [-0.003218989002323255, -0.000654135580220034]).

Retrain flag from drift: False.
