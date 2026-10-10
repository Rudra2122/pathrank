# Model card: PathRank (synthetic, feature store v1)

Registry: ranker v1, two-tower v1; gate: PROMOTED (first registered model).
Lineage: features a7380f01e8c0, git nogit, text encoder tfidf.

## Intended use
Rank open job postings for an early-career job seeker and decide whether to notify them. Not for screening candidates, setting pay, or any decision about a person's eligibility.

## Data
20,000 users, 30,000 jobs, 163,918 applications; time split start 2012-04-01, ctx 2012-06-05, val 2012-06-20, test 2012-06-25, end 2012-06-30.
Synthetic data: metrics validate the pipeline only.

## Performance (held-out final window)
| model | ndcg@10 | recall@100 |
|---|---|---|
| popularity | 0.0067 | 0.0618 |
| location_popularity | 0.0583 | 0.2051 |
| two_tower_only | 0.1215 | 0.4283 |
| ranker_v1 | 0.1609 | 0.4555 |
| ranker_v2 | 0.2056 | 0.5034 |

## Fairness
Non-metro jobs: 51.1% of catalog. Exposure vs NDCG trade-off:
| min non-metro share | ndcg@10 | non-metro exposure |
|---|---|---|
| 0.0 | 0.2056 | 40.6% |
| 0.1 | 0.2048 | 41.8% |
| 0.2 | 0.2032 | 43.7% |
| 0.3 | 0.2020 | 45.7% |
| 0.4 | 0.2007 | 48.4% |

NDCG@10 by degree type: Associate's 0.216, Bachelor's 0.204, High School 0.196, Master's 0.201, None 0.214, PhD 0.204, Vocational 0.210

## Limitations
- Offline evaluation on logged applications: exposure bias of the original site is baked into labels.
- Notification targeting is predictive, not causal (no randomized sends in the data).
- LightGCN only covers users/jobs already in the interaction graph.
- Text embeddings can encode demographic proxies through job titles and majors; monitor by group.

## Monitoring
PSI drift on ranker features each run; retraining is triggered when drift exceeds 0.25 on more than 2 features.