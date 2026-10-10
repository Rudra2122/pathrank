"""GPU benchmark script: runs anywhere, records skips instead of crashing; FP16 TensorRT source exported."""
import subprocess
import sys

from pathrank.utils import load_json


def test_gpu_benchmark_runs_and_records(built):
    r = subprocess.run([sys.executable, "scripts/gpu_benchmark.py", "--config", "configs/ci.yaml",
                        "--batches", "1,32", "--iters", "5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    g = load_json(built.art_dir / "gpu_benchmark.json")
    for tower in ("user_tower", "item_tower"):
        assert {"torch_cpu", "ort_cpu"} <= set(g[tower])
        assert g[tower]["ort_cpu"]["parity"]["max_abs_diff"] < 1e-4
    import torch
    if not torch.cuda.is_available():
        assert "user_tower.tensorrt" in g["skipped"]


def test_tensorrt_sources_exported(built):
    repo = built.art_dir / "model_repository" / "user_tower_trt"
    assert (repo / "model_int32.onnx").exists() and (repo / "model_int32_fp16.onnx").exists()
    e = load_json(built.art_dir / "results.json")["export"]
    fp16 = e["user_tower_trt_fp16_source"]
    assert "cpu_check" in fp16 or fp16["max_abs_diff"] < 1e-2


def test_training_is_deterministic_on_cpu(built):
    import numpy as np
    from pathrank.features.pipeline import load_store
    from pathrank.retrieval.lightgcn import train_lightgcn
    s = load_store(built)
    e = s.apps[s.apps.ts < s.cutoffs["val"]]
    a, b = train_lightgcn(e, built.lightgcn, 0), train_lightgcn(e, built.lightgcn, 0)
    assert np.array_equal(a.U, b.U) and np.array_equal(a.I, b.I)
