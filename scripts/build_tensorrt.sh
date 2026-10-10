#!/usr/bin/env bash
# Build the TensorRT engine for Triton's user_tower_trt model. NVIDIA GPU required.
# The engine must be built with the same TensorRT version as the Triton container that serves
# it, so trtexec runs INSIDE the matching Triton image.
#
#   TRITON_TAG=24.08 bash scripts/build_tensorrt.sh artifacts/careerbuilder/v1/model_repository
#   (pick a current tag from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
#
# Precision: TensorRT <= 10 builds the FP32 graph with --fp16. TensorRT >= 11 is always strongly
# typed and has no --fp16, so it compiles the FP16 graph (model_int32_fp16.onnx) as-is.
# PRECISION=fp32 builds an FP32 engine on either.
set -euo pipefail
REPO=$(realpath "${1:?path to model_repository}")
TAG=${TRITON_TAG:-24.08}
PRECISION=${PRECISION:-fp16}
MAXB=${MAX_BATCH:-1024}
OPTB=${OPT_BATCH:-64}

cfg="$REPO/user_tower/config.pbtxt"
HIST=$(python3 -c "import re;t=open('$cfg').read();print(re.search(r'name: \"tok\"[^}]*dims: \[ (\d+) \]',t).group(1))")
TEXT=$(python3 -c "import re;t=open('$cfg').read();print(re.search(r'name: \"text\"[^}]*dims: \[ (\d+) \]',t).group(1))")
shapes() { echo "state:$1,city:$1,degree:$1,major:$1,tok:${1}x${HIST},text:${1}x${TEXT},num:${1}x6"; }

mkdir -p "$REPO/user_tower_trt/1"
cat > "$REPO/user_tower_trt/_build.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
T=/usr/src/tensorrt/bin/trtexec
SRC=model_int32.onnx; FLAG=""
if [ "$PRECISION" = fp16 ]; then
  if \$T --help 2>&1 | grep -q -- "--fp16"; then FLAG="--fp16"; else SRC=model_int32_fp16.onnx; fi
fi
echo "trtexec: building from \$SRC \$FLAG"
\$T --onnx=/models/user_tower_trt/\$SRC --saveEngine=/models/user_tower_trt/1/model.plan \\
    --minShapes=$(shapes 1) --optShapes=$(shapes "$OPTB") --maxShapes=$(shapes "$MAXB") \\
    \$FLAG --memPoolSize=workspace:2048 --skipInference
EOS
docker run --rm --gpus all -v "$REPO":/models "nvcr.io/nvidia/tritonserver:${TAG}-py3" bash /models/user_tower_trt/_build.sh
rm -f "$REPO/user_tower_trt/_build.sh" "$REPO/user_tower_trt/BUILD_ON_GPU.txt"
echo "engine: $REPO/user_tower_trt/1/model.plan"
