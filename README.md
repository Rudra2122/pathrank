# PathRank

Job recommendations and notifications for early-career job seekers: a smaller version of the
problem a growth-relevance team at a jobs marketplace runs. It retrieves jobs from four
sources, ranks them with LightGBM, re-ranks for exposure fairness, decides whether and when
to notify each user, and has a LangGraph agent write the notification behind hard guardrails.

```
users + jobs + applications (CareerBuilder, Kaggle)
  -> data validation gate (22 contract checks; errors stop the pipeline)
  -> feature pipeline: versioned parquet + text embeddings, "as-of" contexts (no future leakage)
  -> Feast feature store: point-in-time offline snapshots, SQLite/Redis online store
  -> STAGE 1 retrieval, ~460 candidates/user, merged from:
       two-tower (PyTorch, content features, in-batch softmax + logQ) + FAISS
       LightGCN on the user-job graph (PyTorch sparse)
       text match (sentence-transformers / TF-IDF+SVD) for cold-start users
       location popularity (city -> state -> national)
  -> STAGE 2 LightGBM LambdaRank (v1: retrieval features, v2: + context features), SHAP reasons
  -> STAGE 3 exposure re-rank (FA*IR-style prefix constraint for non-metro jobs), NDCG cost sweep
  -> notifications: apply-in-7-days classifier + days-to-return regression -> send / when / cap
  -> LangGraph agent: profile -> recs -> draft -> guardrails (retry/block) -> LLM judge
  -> ONNX export -> Triton model repository (ONNX towers, FIL ranker, TensorRT engine)
  -> FastAPI (+ Prometheus) reading features from the online store, models in-process or on Triton
  -> MLflow tracking + registry (champion/challenger gate), drift-triggered retraining, canary deploys
  -> evalwright SDKs (sdk/): tracing + evals for the agent and the API, Python + TypeScript, one wire protocol
```


## Results on real data (OLX Jobs Interactions, Kaggle)

65.5M job-board events -> 46.4M user-job pairs, 2.57M users, 176K jobs. Trained on one Colab A100.
Full reports: `results/RESULTS_olx_full.md`, `results/RESULTS_olx_300k.md`.

| | full data | 300K-user sample |
|---|---|---|
| Ranker v2 NDCG@10 vs popularity | **9.4x** | 9.3x |
| v2 vs v1 (paired bootstrap 95% CI) | +10.2% [+0.0005, +0.0026] | +40.2% [+0.0039, +0.0064] |
| Simulated 50/50 A/B, v2 vs v1 | p = 0.14 (underpowered) | p < 0.001 |
| Two-tower recall@100 | 0.039 | 0.010 |
| Notification AUC (recency baseline) | 0.865 (0.815) | 0.864 (0.815) |
| Sends needed to reach 80% of applicants | 39% of users | 40% of users |
| Feast online lookup p50 | 0.68 ms (2.57M users) | 1.0 ms |
| LightGCN on A100 | 34.8M edges, ~47 s/epoch | 4.1M edges, ~4 s/epoch |

Known limits: interactions-only data (no job text or locations), so cold-start users score
NDCG 0.0009 vs 0.0242 warm; only ~17% of test interactions reach the ranker (retrieval is the ceiling).
A label leak from inferred job end dates was found via PSI drift monitoring and fixed (see `job_window`).

## Data, honestly

**Real data (use for every number you publish):** the
[CareerBuilder Job Recommendation Challenge](https://www.kaggle.com/c/job-recommendation) on
Kaggle: users (degree, major, location, experience, work history), job postings with
descriptions and posting windows, and real applications. `scripts/download_data.sh` prints
exact row counts after download; the run writes them into `RESULTS.md`.

**About "10M+ records":** I couldn't find a public job-application dataset at 10M+ interactions
that you can download and cite. CareerBuilder is the right one; across its four files it's a
few million rows, of which about 1.6M are applications. So this repo has two modes:

| mode | config | purpose |
|---|---|---|
| CareerBuilder | `configs/careerbuilder.yaml` | all accuracy metrics, all resume numbers |
| synthetic, 10M+ applications | `configs/synthetic_10m.yaml` | scale/throughput benchmark only |
| synthetic, small | `configs/synthetic_small.yaml`, `configs/ci.yaml` | smoke tests, CI |

The synthetic generator writes byte-for-byte the same TSV schema, so it exercises the same
loader. It plants the structure the models then find, so **never quote synthetic accuracy**.
"Benchmarked the pipeline at 12M applications" is fine; "NDCG X on 12M applications" is not.

## Quickstart (MacBook)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make smoke                     # whole pipeline on small synthetic data, a few minutes
make test                      # 30 tests: leakage, determinism, feature-store parity + concurrency, Triton protocol, registry gate, API tracing
make sdk-test                  # evalwright: Python + TypeScript suites and the cross-language contract

pip install kaggle             # token in ~/.kaggle/kaggle.json, accept the competition rules once
make data                      # downloads + unzips to data/raw/careerbuilder
make weekend1                  # validation, Feast, two-tower, LightGCN, rankers, offline eval
make weekend2                  # notification models
make weekend3                  # ONNX/Triton export, agent eval, drift, registry gate, reports, TF baseline
make serve                     # http://localhost:8080/recommend/<UserID>  (also /features, /notify)
make mlflow-ui                 # runs, metrics, registered models + aliases
make triton-cpu                # Triton in Docker on CPU (ONNX towers + FIL ranker), then:
make serve-triton              # API with model inference on Triton
make stack                     # Redis + Triton + API + Prometheus via docker compose
```

If your Mac struggles: `python scripts/run_pipeline.py --config configs/careerbuilder.yaml --set data.max_users=100000`.
For a real agent eval: `export ANTHROPIC_API_KEY=...` then `--stages agent` (default model in
`configs`, change `agent.model` if you like). Without a key the agent uses a deterministic
template, which makes the guardrail pass rate trivially 100%; RESULTS.md says so.

## Outputs

- `RESULTS.md` (repo root for real data; `artifacts_*/.../RESULTS.md` for synthetic): every
  table below is generated from `results.json`, nothing is typed by hand.
- `artifacts/careerbuilder/v1/analytics.html`: cohort retention, daily activity, notification targeting curve.
- `drift_psi.csv`, `drift_report.html` (if Evidently installed), `agent_eval.jsonl`, model files.

## evalwright: the LLM evaluation and observability SDKs (`sdk/`)

PathRank's agent and API are instrumented with **evalwright**, a tracing and evaluation SDK built in
this repo in two languages that speak one protocol. Details in [`sdk/README.md`](sdk/README.md).

| | Python (`sdk/python`) | TypeScript (`sdk/typescript`) |
|---|---|---|
| API | `init`, `span`, `@traced`, `log`, `Eval`, scorers, `profiled` | `init`, `span`, `traced`, `log`, `Eval`, scorers |
| runtime deps | none (stdlib only) | none |
| typing | mypy `--strict` | `strict` + `exactOptionalPropertyTypes` |
| runtimes tested | CPython 3.10, 3.11, 3.12, 3.13 | Node 18, 20, 22 (packed tarball), 22/24 (unit) |
| testing framework | `evalwright.testing` + auto-registered pytest plugin | `evalwright/testing` |
| tooling | CLI: `run` (CI gates), `compare`, `ls`, `serve` (collector with fault injection); overhead profiler with CI budget | publint + are-the-types-wrong; ESM + CJS smoke test |
| tests | 90, 94% coverage | 53, 97% line coverage |

- **Resilient by contract** ([`sdk/spec/PROTOCOL.md`](sdk/spec/PROTOCOL.md)): bounded drop-oldest queues,
  retries with full jitter and `Retry-After`, circuit breaker, content-derived idempotency keys so a retry
  that already landed is deduped, flush-with-timeout on exit, total/bounded serialization, secret
  redaction, and "never raise into or block the host app". A cross-language contract test delivers both
  SDKs' spans through a 30% injected failure rate with zero loss and zero duplicates, schema-validated.
- **Cross-language by design**: identical sampling decisions (CRC-32 of the trace id), identical eval case
  ids, and a shared experiment format, so the Python CLI compares TypeScript eval runs. Parity tests pin the
  rules with values generated by the other SDK.
- **In PathRank**: the agent stage is an `evalwright.Eval` (scores `delivered`, `first_try`,
  `judge_quality` with bootstrap CIs, compared case-by-case with the previous pipeline run, gated); every
  LangGraph node is a span; the API traces each request by route template with the recommender's stages
  nested inside. Set `EVALWRIGHT_EXPORT=otlp:http://collector:4318/v1/traces` to ship API traces to any
  OpenTelemetry backend.
- **Bugs it found here**: running agent evals concurrently exposed that Feast's SQLite online store shares
  one connection across threads (spurious "user missing" and `sqlite3.InterfaceError` under concurrent
  requests, which the live API would hit too), and that the frequency cap's check-then-record wasn't atomic.
  Both are fixed and locked in by regression tests. Building the TypeScript package caught a separately
  bundled `testing` entry point holding its own copy of the global tracer; global state now lives on
  `globalThis` (the dual-package-hazard fix OpenTelemetry uses), verified by loading ESM and CJS together.

## Running on a Colab GPU (A100 / L4 / T4)

Open `colab_gpu.ipynb` in Colab, pick an A100 runtime, and run the cells top to bottom. It:
checks the GPU, installs (ONNX Runtime GPU + TensorRT), runs a synthetic smoke test, downloads
CareerBuilder, runs the full pipeline with **training on the GPU**, then runs
`scripts/gpu_benchmark.py` and writes a GPU table into `RESULTS.md`. Results are zipped to Drive.

The benchmark compares the user and item towers on: PyTorch CPU, ONNX Runtime CPU, PyTorch CUDA
(FP32 and FP16 autocast), ONNX Runtime CUDA, ONNX Runtime + TensorRT (FP16), and a native
TensorRT FP16 engine. It reports p50/p99 latency per batch size, throughput, and parity against
PyTorch FP32. Anything unavailable is skipped with the reason recorded; nothing is faked.

- Triton does not run in Colab (no Docker). Use a GPU VM or RunPod/Vast.ai for `scripts/triton_perf.sh`.
- The GPU name in the results comes from the device. If Colab hands you an L4, the table says L4.
- A TensorRT engine is tied to the GPU model and TensorRT version that built it.
- GPU training isn't bit-reproducible (CUDA scatter-adds use atomics); CPU training is, and a test enforces it.

## Production serving: feature store, Triton, TensorRT, MLOps

What runs where, and what was verified while building this repo:

| piece | what it does | verified here |
|---|---|---|
| **Feast feature store** | `user_dynamic` + `job_dynamic` feature views; offline Parquet snapshots at each cutoff; point-in-time retrieval for training; materialized online store | yes: PIT and online values equal the feature pipeline exactly (max diff 0.0), SQLite **and Redis** |
| **Online feature serving** | API fetches user features per request from the online store (`/features/{id}` exposes them); job features refreshed with the catalog | yes: full recommender gives identical top-10 from online vs offline features; 0.4-0.7 ms p50 fetch |
| **Triton Inference Server** | model repository: `user_tower`, `item_tower` (ONNX Runtime backend, dynamic batching), `ranker` (FIL backend, LightGBM on GPU) | configs parsed against Triton's ModelConfig protobuf; client + exported models + configs tested end to end against a KServe-v2 test server that enforces the configs. **Not run on real Triton here** (no Docker daemon/registry access): run `make triton-cpu` on your Mac |
| **TensorRT** | FP16 engine for the user tower: built by `trtexec` inside the matching Triton image for serving, and by the TensorRT Python API in `scripts/gpu_benchmark.py` (Colab); also ONNX Runtime's TensorRT provider | FP32 and FP16 source graphs checked (1e-7 / 2e-3). Build recipe handles TensorRT 10 (FP16 flag) and 11 (always strongly typed, FP16 graph); API names checked against TensorRT 11.4's bindings. **Engines need an NVIDIA GPU**: Colab notebook, or `scripts/gpu_vm.sh` + `make tensorrt` |
| **MLOps** | data validation gate; MLflow registry with lineage tags; champion/challenger promotion gate (paired bootstrap on the same held-out users); model card; drift flag; weekly retrain workflow; Cloud Run canary + promote/rollback; registry-backed serving (`MODEL_SOURCE=registry`) | yes: identical challenger rejected, weaker champion replaced, registry-served recs equal local |

Your M3 has no NVIDIA GPU, so TensorRT can't run on it; Triton runs on it in Docker on CPU.
Put only measured GPU numbers on a resume, from `triton_perf/*.csv` after a T4 run (about an hour).

## Design decisions worth defending in an interview

1. **Content towers, not ID embeddings.** On job boards most postings in any test window are
   new. An ID-based model can't score a job it has never seen. Both towers use location, title
   tokens and description embeddings, so a job posted this morning is retrievable.
2. **LightGCN is kept, and its limit is measured.** It only scores users and jobs already in
   the graph; RESULTS.md reports its catalog and user coverage. Its value shows up as a ranker
   feature rather than a standalone retriever. That's a finding, not a failure.
3. **As-of contexts everywhere.** `AsOf(t)` only sees applications before `t`; popularity,
   recency and user history are recomputed per cutoff. Retriever trains as-of `ctx`, ranker
   as-of `val`, evaluation as-of `test`. A test checks that nothing from the label window leaks
   into features.
4. **Metrics count every user.** If a user's application was never retrieved, their NDCG is 0.
   Candidate recall is reported separately so you can see where the funnel leaks.
5. **Baselines that are hard to beat.** Global popularity is a straw man for local jobs; the
   honest baseline is location popularity. Lead with the lift over that one.
6. **Fairness is a sweep, not a switch.** Non-metro jobs get a minimum share of every top-k
   prefix; RESULTS.md shows exposure gained vs NDCG lost at each level. User-side NDCG by degree
   type is reported separately.
7. **Notifications: targeting, not uplift.** There are no randomized sends in the data, so the
   claim is "reach 80% of next-week applicants by notifying N% of users", never "caused X more
   applications". Also compare against the recency-only heuristic: if the model barely beats it,
   say so.
8. **Guardrails are code, not prompts.** Job IDs and every quoted title must come from the
   recommended set; no pay figures (not in the data), no contact info; failures retry once then
   block. The eval includes adversarial messages the guardrail must catch.
9. **Same code offline and online.** `Recommender` + `AsOf` produce features for the batch eval
   and for `/recommend`, so training/serving skew can't come from two implementations.
10. **Feature store = storage, not a second implementation.** Feature logic lives once (`AsOf`);
    Feast stores and serves its output. Parity tests prove offline (PIT) and online values
    match the pipeline exactly, which is the actual skew guarantee.
11. **Triton configs are tested, not hoped.** Scalar inputs are declared `dims: [1]` + reshape,
    so clients must send `[B, 1]`; a test proves a `[B]` request is rejected. TensorRT gets an
    int32-input graph because int64 support differs across TensorRT versions.
12. **Version-aware TensorRT.** TensorRT 11 removed the FP16 builder flag (networks are always
    strongly typed), so precision has to live in the graph. The export writes both an FP32 and an
    FP16 int32-input graph, and the builders pick by TensorRT version.
13. **Determinism as a test.** LightGCN's gradients through repeated-index gathers were
    nondeterministic on CPU, enough to move ranker NDCG in the 3rd decimal between identical runs.
    Fixed (symmetric-adjacency custom backward + `index_select`) and locked in with a test.
14. **Promotion is a statistical decision.** A new model replaces the champion only when the
    paired-bootstrap CI of its NDCG@10 gain over the champion, on the same users and candidates,
    is above zero. Rollback is moving the `champion` alias back.

## What the smoke run already found (synthetic, pipeline-validation only)

- PyTorch's default `N(0,1)` embedding init was hurting the two-tower model; the Keras baseline
  (±0.05 init) beat it on identical inputs. Switching to `std=0.05` raised two-tower recall@100
  from 0.32 to 0.43 and ranker NDCG@10 from 0.177 to 0.206. Check whether this holds on
  CareerBuilder; it's a good "how I debugged a model" story either way.
- LightGCN needed lr 1e-2 and ~15+ epochs before its BPR loss moved at all.
- The notification classifier only slightly beat a pure recency heuristic on synthetic data. If
  that holds on real data, the honest bullet is about the regression and send timing.

## Scale benchmark (synthetic 10M+, measured on this build machine)

See `artifacts_10m/synthetic/v1/benchmark.json` after `make bench10m`. Numbers from the run
used to build this repo (2 vCPU / 7 GB cloud container, so your M3 will be faster):

| measurement | value |
|---|---|
| dataset | 1.5M users, 1.0M jobs, **11.3M applications**, 3.0M work-history rows (16.8M records) |
| generate synthetic TSVs | 245 s (peak 5.3 GB) |
| build feature store (load, clean, TF-IDF+SVD on 1M job texts, parquet) | 412 s |
| load feature store / build as-of context | 4.8 s / 3.3 s |
| two-tower training | 5.9K pairs/s, ~6.5 min/epoch over 2.3M pairs |
| LightGCN (9.0M edges, 1.9M nodes, dim 32) | 11.3 s/full-graph step, ~26 min/epoch at batch 65,536 |
| FAISS over 293K open jobs | 1.0K queries/s flat, 4.9K queries/s HNSW |
| batch retrieval + features | 153 users/s, ~496 candidates/user |
| LightGBM ranker scoring | 559K rows/s |
| **single-user request (features + 3 retrievers + rank)** | **92 ms p50, 107 ms p95** |
| peak memory | 5.8 GB |

Two scaling bugs this benchmark caught: per-request latency was 1.44 s p50 until the FAISS
indexes, location tables and per-user history offsets were cached per context (now 92 ms);
and LightGCN ran out of memory until the adjacency was built directly as CSR and the
default batch was raised (full-graph propagation runs every step, so fewer, larger steps).

## Resume bullets (fill X only from `RESULTS.md` on CareerBuilder)

- Lifted NDCG@10 X% over a location-popularity baseline on ~1.6M real job applications with a
  3-stage recommender: content two-tower + LightGCN + text retrieval with FAISS, LightGBM
  LambdaRank with SHAP explanations, and an exposure-fairness re-rank (X% more non-metro
  exposure for X% NDCG cost)
- Reached 80% of next-week applicants by notifying X% of users with an apply classifier (AUC X)
  and a return-time model (MAE X days); notification copy written by a LangGraph agent behind
  code-level guardrails, served with FastAPI on Cloud Run
- Benchmarked the pipeline at 10M+ synthetic applications: X ms p95 per-user ranking latency
- Served user features from a Feast online store (Redis) with point-in-time-correct training
  snapshots, proving zero training/serving skew; X ms p95 feature fetch
- Cut user-tower latency X× (p99 X ms → X ms) by exporting to ONNX and compiling a TensorRT FP16
  engine on an A100, with embedding parity (min cosine X) against FP32 PyTorch
  *(only from colab_gpu.ipynb / gpu_benchmark.py output)*
- Deployed the two-tower and LightGBM ranker on Triton Inference Server (ONNX Runtime + FIL
  backends, dynamic batching) and a TensorRT FP16 engine: X× throughput / X ms p99 vs PyTorch on a T4
  *(only after running scripts/triton_perf.sh on a GPU)*
- Built the MLOps loop: data-validation gate, MLflow registry with a statistical
  champion/challenger promotion gate, drift-triggered weekly retraining, Cloud Run canary deploys
- Built evalwright, a zero-dependency LLM tracing and evaluation SDK in Python and TypeScript sharing one
  wire protocol: async-context propagation, batching exporter with retries/jitter/circuit breaker and
  idempotent delivery (zero loss, zero duplicates through 30% injected failures), eval runner with
  bootstrap CIs and regression gates, pytest plugin, span profiler, OTLP export
- Shipped it like a product: mypy --strict / strict TS, 140+ tests across Python 3.10-3.13 and Node 18-24,
  cross-language contract tests, installed-wheel and packed-tarball smoke tests, CI/CD releasing to
  PyPI (trusted publishing) and npm (provenance); dogfooding it surfaced 2 concurrency bugs in PathRank

Write whatever comes out, including weak results and surprises.

## Layout

```
pathrank/
  data/        careerbuilder.py (loader), synthetic.py (generator)
  features/    pipeline.py (store, AsOf), text.py (encoders)
  retrieval/   two_tower.py, lightgcn.py, simple.py (text/popularity), index.py (FAISS)
  ranking/     features.py, ranker.py (LambdaRank), explain.py (SHAP)
  rerank/      fairness.py
  notify/      models.py, policy.py (cap, send time, sponsored-slot EV)
  agent/       graph.py (LangGraph), guardrails.py, llm.py (Anthropic or template + judge)
  eval/        metrics.py, ab.py (replayed A/B)
  featurestore/ repo.py (Feast definitions), materialize.py, online.py (OnlineContext)
  inference/   export.py (ONNX + Triton repo + TensorRT source), backends.py (torch/onnx/triton)
  mlops/       validate.py (data gate), registry.py (MLflow registry, promotion gate, model card)
  monitoring/  drift.py      analytics/ report.py      serve/ app.py, service.py
  recommender.py  config.py  device.py (cuda/cpu/mps)  tracking.py (MLflow)
sdk/           evalwright SDKs: spec/ (wire protocol), python/, typescript/
scripts/       run_pipeline.py, benchmark_scale.py, gpu_benchmark.py, download_data.sh,
               deploy_cloud_run.sh (canary), build_tensorrt.sh, triton_perf.sh, gpu_vm.sh
colab_gpu.ipynb  Colab GPU runbook (training on GPU + TensorRT benchmark)
deploy/        docker-compose.yml (Redis + Triton + API + Prometheus)
.github/       ci.yml, retrain.yml, sdk-python-ci/-release, sdk-typescript-ci/-release, sdk-contract
tf_baseline/   two_tower_keras.py
tests/         unit, end-to-end, feature-store parity, Triton protocol (triton_stub.py), registry gate, API
```

## Notes

- LightGCN is plain PyTorch sparse ops (same math as PyG's `LGConv`) to avoid compiled
  extensions on Apple Silicon.
- TensorFlow Recommenders breaks on Keras 3, so the TF baseline is plain Keras with the same
  in-batch softmax objective. On Apple Silicon install `tensorflow-macos` (see `requirements-extra.txt`).
- The data is from 2012; the API serves "as of" the test cutoff (`SERVE_CUTOFF=end` for the last day).
- Triton image tag defaults to `24.08`; use a current tag from NVIDIA NGC, and build the
  TensorRT engine with the same tag you serve with.
- The 10M benchmark measures the in-process path; materializing 1.5M users into the online
  store wasn't run at that scale here (7 GB box).
- Things this project can't show: years of production ML experience and cross-functional work.
