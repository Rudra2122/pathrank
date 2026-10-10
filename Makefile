PY ?= python
CFG ?= configs/careerbuilder.yaml

install:        ## core deps
	$(PY) -m pip install -r requirements.txt
data:           ## download CareerBuilder from Kaggle
	bash scripts/download_data.sh
weekend1:       ## data -> features -> validation -> Feast -> two-tower + LightGCN -> LightGBM ranker -> eval
	$(PY) scripts/run_pipeline.py --config $(CFG) --stages data,features,validate,featurestore,retrieval,rank
weekend2:       ## notification models (+ SHAP / fairness / A/B already ran in rank)
	$(PY) scripts/run_pipeline.py --config $(CFG) --stages notify
weekend3:       ## ONNX/Triton export, agent eval, monitoring, registry + promotion gate, report, TF baseline
	$(PY) scripts/run_pipeline.py --config $(CFG) --stages export,agent,monitor,register,report
	$(PY) tf_baseline/two_tower_keras.py --config $(CFG)
all:
	$(PY) scripts/run_pipeline.py --config $(CFG) --stages all
smoke:          ## whole pipeline on small synthetic data, a few minutes
	$(PY) scripts/run_pipeline.py --config configs/synthetic_small.yaml --stages all
olx:            ## REAL data from Kaggle: OLX Jobs Interactions (300K-user sample; needs kaggle CLI + token)
	$(PY) scripts/run_pipeline.py --config configs/olx.yaml --stages all
olx-full:       ## all 65.5M OLX events (high-RAM machine / Colab A100)
	$(PY) scripts/run_pipeline.py --config configs/olx_full.yaml --stages all
xing:           ## REAL data: XING RecSys 2016 (5.7M positive pairs; downloads from GitHub, no token)
	$(PY) scripts/run_pipeline.py --config configs/xing.yaml --stages all
bench10m:       ## 10M+ application scale benchmark (synthetic)
	$(PY) scripts/benchmark_scale.py --config configs/synthetic_10m.yaml
test:
	$(PY) -m pytest -q -p no:warnings tests
sdk-test:       ## evalwright SDKs: Python suite, TypeScript suite + build, cross-language contract
	cd sdk/python && $(PY) -m pytest -q -p no:warnings && $(PY) -m mypy
	cd sdk/typescript && npm ci && npm run typecheck && npm test && npm run build && npm run check:package
	cd sdk/python && $(PY) -m pytest -q -p no:warnings tests/test_contract.py
REPO = artifacts/careerbuilder/v1/model_repository
gpu-bench:      ## GPU inference benchmark (torch / ORT-CUDA / ORT-TensorRT / native TensorRT); skips what's missing
	$(PY) scripts/gpu_benchmark.py --config $(CFG)
tensorrt:       ## NVIDIA GPU only: build the TensorRT engine for the user tower
	bash scripts/build_tensorrt.sh $(REPO)
triton-perf:    ## NVIDIA GPU: serve with Triton, measure with perf_analyzer
	bash scripts/triton_perf.sh $(REPO)
triton-cpu:     ## Triton on CPU (Mac/Linux, Docker): ONNX towers + FIL ranker
	docker run --rm -p 8000:8000 -p 8002:8002 -v $(PWD)/$(REPO):/models nvcr.io/nvidia/tritonserver:$${TRITON_TAG:-24.08}-py3 \
	  tritonserver --model-repository=/models --model-control-mode=explicit --load-model=user_tower --load-model=item_tower --load-model=ranker
serve-triton:   ## API with towers + ranker on Triton (run triton-cpu first)
	TRITON_URL=localhost:8000 PATHRANK_CONFIG=$(CFG) uvicorn pathrank.serve.app:app --port 8080
feast-ui:       ## browse the feature repo
	cd artifacts/careerbuilder/v1/feature_repo && feast ui
stack:          ## Redis + Triton + API + Prometheus
	docker compose -f deploy/docker-compose.yml up
serve:
	PATHRANK_CONFIG=$(CFG) uvicorn pathrank.serve.app:app --port 8080
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
.PHONY: olx olx-full xing sdk-test gpu-bench install data weekend1 weekend2 weekend3 all smoke bench10m test serve mlflow-ui tensorrt triton-perf triton-cpu serve-triton feast-ui stack
