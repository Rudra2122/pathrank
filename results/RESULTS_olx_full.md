# PathRank results (olx, feature store v1 / c60440b31e56)

Users 2,570,548 | jobs 176,644 | applications 46,441,920 | text encoder: none (16-d)

Time split: start 2020-02-12, ctx 2020-02-20, val 2020-02-22, test 2020-02-23, end 2020-02-25

Applications per split: retriever_train 7,690,291, ranker_train 3,742,338, test 7,908,646

Data validation: 22 checks, 0 errors, 4 warnings (see data_validation.csv).

## Feature store (Feast) and online feature serving

Online store: sqlite; 2,570,548 users and 160,852 open jobs materialized at 2020-02-23. Offline snapshots: 2,574,548 user rows, 490,772 job rows.

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

Online user-feature fetch: p50 0.68 ms, p95 0.72 ms.

## Stage 1: retrieval (test window, each source alone)

| source | recall@100 | ndcg@10 | candidate recall |
|---|---|---|---|
| popularity | 0.0090 | 0.0018 | 0.0090 |
| lp | 0.0059 | 0.0018 | 0.0059 |
| txt | 0.0009 | 0.0001 | 0.0012 |
| tt | 0.0387 | 0.0083 | 0.0744 |
| gcn | 0.0653 | 0.0076 | 0.1243 |
| union_candidates | 0.0398 | 0.0045 | 0.1678 |
Training device: two-tower on cuda:NVIDIA A100-SXM4-40GB (200,226 pairs/s), LightGCN on cuda:NVIDIA A100-SXM4-40GB (466.2 s).


Avg candidates/user: 662. Test applications to jobs with zero history before the cutoff: 0.1%. LightGCN can score 95.4% of the open catalog and 64.9% of eval users.

(lp = location popularity, txt = text match, tt = two-tower, gcn = LightGCN)

## Stage 2: ranking (test window)

| model | ndcg@10 | map@10 | recall@10 | recall@100 | hit@10 |
|---|---|---|---|---|---|
| popularity | 0.0018 | 0.0008 | 0.0017 | 0.0090 | 0.0123 |
| location_popularity | 0.0018 | 0.0008 | 0.0017 | 0.0059 | 0.0123 |
| two_tower_only | 0.0086 | 0.0041 | 0.0086 | 0.0387 | 0.0453 |
| ranker_v1 | 0.0154 | 0.0071 | 0.0153 | 0.0713 | 0.0831 |
| ranker_v2 | 0.0169 | 0.0075 | 0.0176 | 0.0884 | 0.0920 |

NDCG@10 lift, ranker v2 vs popularity: **843.1%**; vs location popularity: **843.1%**.

Cold users (no history): ndcg@10 0.0009 (n=3114); warm: 0.0242 (n=6886).

### Replayed A/B, ranker v1 (retrieval features) vs v2 (+ context features)

Hash-split arms: diff 0.0020 ndcg@10, 95% CI [-0.0005, 0.0048], p=0.136. Paired (same users, both rankers): diff 0.0016, CI [0.0005, 0.0026], relative lift 10.2%.

### SHAP (mean |value|, ranker v2)

| feature | mean abs SHAP |
|---|---|
| job_pop_7d | 0.8470 |
| title_overlap | 0.5903 |
| user_days_since_last | 0.4034 |
| txt_rank | 0.2448 |
| gcn_score | 0.2221 |
| tt_score | 0.1890 |
| gcn_rank | 0.1731 |
| txt_score | 0.1702 |
| job_age_days | 0.1335 |
| user_n_apps | 0.1255 |

Example "why this job" reasons:

- user 0: *job 85511* -> getting a lot of applications this week; people with similar applications applied here
- user 0: *job 147324* -> getting a lot of applications this week; people with similar applications applied here
- user 0: *job 93425* -> getting a lot of applications this week; people with similar applications applied here
- user 963: *job 31871* -> getting a lot of applications this week; people with similar applications applied here
- user 963: *job 140157* -> getting a lot of applications this week; people with similar applications applied here
- user 963: *job 119019* -> people with similar applications applied here; getting a lot of applications this week

## Stage 3: fairness re-rank (non-metro job exposure)

Non-metro jobs are 0.0% of the open catalog and 0.0% of test applications.

| min non-metro share per prefix | ndcg@10 | non-metro exposure |
|---|---|---|
| 0.0 | 0.0169 | 0.0% |
| 0.1 | 0.0169 | 0.0% |
| 0.2 | 0.0169 | 0.0% |
| 0.3 | 0.0169 | 0.0% |
| 0.4 | 0.0169 | 0.0% |

User-side: ndcg@10 by degree type

| degree | ndcg@10 | users |
|---|---|---|
| None | 0.0169 | 10000 |

## Inference export (ONNX / Triton / TensorRT)

ONNX vs PyTorch max abs diff: user tower 1.79e-07, item tower 8.94e-08, TensorRT source graph 1.79e-07. Top-10 lists identical with ONNX towers: 79.9%. Triton configs validated against the ModelConfig schema: item_tower, ranker, user_tower, user_tower_trt.

| user tower, CPU | batch 1 p50 (ms) | batch 1024 p50 (ms) |
|---|---|---|
| torch | 0.413 | 2.135 |
| onnxruntime | 0.067 | 2.398 |

GPU numbers (Triton + TensorRT) come from scripts/triton_perf.sh on an NVIDIA GPU.


## Notifications

Apply-within-2-days classifier: ROC-AUC 0.8652 (recency-only baseline 0.8148), PR-AUC 0.7714 (base rate 0.3162).

Days-to-return regression (returners within cap): MAE 0.6464 days, RMSE 0.9274; median baseline MAE 0.7105.

Targeting: reaching 80% of users who apply within 2 days takes notifying 39.2% of users (60.8% fewer sends than notifying everyone). This is targeting, not causal uplift; the data has no randomized sends.

## Notification agent (LangGraph)

Backend: template. 50 users: ready 100.0%, blocked 0.0%, mean attempts 1.0000, judge quality 0.8750 (0-1). Frequency cap enforced: True. Adversarial guardrail checks caught: 3/3.

Run through the evalwright SDK (experiment `20261010-053254068-b942`, 926 spans traced, case latency p50 799 ms):

| score | mean | 95% CI | n | vs previous run |
|---|---|---|---|---|
| delivered | 1.0000 | [1.000, 1.000] | 50 | - |
| first_try | 1.0000 | [1.000, 1.000] | 50 | - |
| judge_quality | 0.8750 | [0.875, 0.875] | 50 | - |

Gates: all passed.

Template backend: guardrail pass rate is trivially high. Set ANTHROPIC_API_KEY and re-run `--stages agent` for a real number.

## Monitoring (PSI, ranker-train window vs test window)

2 features drifting (PSI > 0.25), 0 to watch. Top: job_age_days 4.8695, user_days_since_last 0.0371, user_n_apps 0.0344, title_overlap 0.0277, user_cold 0.0277

## Model registry and promotion

MLflow registry: ranker v1, two-tower (ONNX) v1; lineage features c60440b31e56, git nogit. Gate: **promoted to champion** (first registered model).

Retrain flag from drift: False.
