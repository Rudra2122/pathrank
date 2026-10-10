#!/usr/bin/env bash
# Serve the repository with Triton on a GPU and measure it with perf_analyzer.
# Compares: ONNX Runtime (CUDA), ONNX Runtime + TensorRT execution accelerator, native TensorRT plan,
# and the FIL ranker. Results land in $REPO/../triton_perf/*.csv
#
#   TRITON_TAG=24.08 bash scripts/triton_perf.sh artifacts/careerbuilder/v1/model_repository
set -euo pipefail
REPO=$(realpath "${1:?path to model_repository}")
TAG=${TRITON_TAG:-24.08}
OUT=$(dirname "$REPO")/triton_perf
mkdir -p "$OUT"
HIST=$(python3 -c "import re;t=open('$REPO/user_tower/config.pbtxt').read();print(re.search(r'name: \"tok\"[^}]*dims: \[ (\d+) \]',t).group(1))")
TEXT=$(python3 -c "import re;t=open('$REPO/user_tower/config.pbtxt').read();print(re.search(r'name: \"text\"[^}]*dims: \[ (\d+) \]',t).group(1))")
NF=$(wc -l < "$REPO/ranker/features.txt")

docker rm -f pathrank-triton >/dev/null 2>&1 || true
docker run -d --name pathrank-triton --gpus all --shm-size=1g -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$REPO":/models "nvcr.io/nvidia/tritonserver:${TAG}-py3" tritonserver --model-repository=/models
until curl -fs localhost:8000/v2/health/ready; do sleep 2; done; echo "triton ready"

SDK="docker run --rm --net host -v $OUT:/out nvcr.io/nvidia/tritonserver:${TAG}-py3-sdk"
USHAPES="--shape state:1 --shape city:1 --shape degree:1 --shape major:1 --shape tok:$HIST --shape text:$TEXT --shape num:6"
for m in user_tower user_tower_trt; do
  curl -fs "localhost:8000/v2/models/$m/ready" || { echo "skip $m (not loaded)"; continue; }
  for b in 1 64 256; do
    $SDK perf_analyzer -m "$m" -b "$b" $USHAPES --input-data zero --concurrency-range 1:8:2 -f "/out/${m}_b${b}.csv" --measurement-interval 5000
  done
done
$SDK perf_analyzer -m ranker -b 500 --shape "input__0:$NF" --input-data zero --concurrency-range 1:4 -f /out/ranker_b500.csv
docker logs pathrank-triton 2>&1 | tail -20 > "$OUT/server_tail.log"
echo "results in $OUT"
