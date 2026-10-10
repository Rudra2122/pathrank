"""GPU inference benchmark for the user tower (the per-request model) and the item tower.

Backends, each skipped with a recorded reason if unavailable:
  torch_cpu        PyTorch on CPU (baseline)
  ort_cpu          ONNX Runtime on CPU
  torch_cuda       PyTorch on GPU, FP32
  torch_cuda_fp16  PyTorch on GPU, autocast FP16
  ort_cuda         ONNX Runtime, CUDA execution provider
  ort_tensorrt     ONNX Runtime, TensorRT execution provider (FP16, engine cache)
  tensorrt         native TensorRT engine (FP16) built here with the TensorRT Python API from
                   the int32-input ONNX graph (TensorRT <= 10: FP32 graph + FP16 flag; TensorRT >= 11,
                   always strongly typed: the FP16 graph); executed with torch-allocated device buffers

Every timing is end to end from host numpy arrays to host numpy arrays (host->device copies
included for GPU backends), which is what a serving process actually pays.
For every backend: p50/p99 latency per batch size, throughput, and parity against PyTorch
FP32 (max abs diff and min cosine similarity of embeddings). Results go to
artifacts/<source>/<version>/gpu_benchmark.json and into results.json ("gpu"), and RESULTS.md
renders them. The GPU name is recorded from the device, so the table says what you really got.

  python scripts/gpu_benchmark.py --config configs/colab_a100.yaml
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathrank.config import load_config  # noqa: E402
from pathrank.inference.export import ITEM_INPUTS, USER_INPUTS  # noqa: E402
from pathrank.utils import load_json, log, save_json  # noqa: E402


def env_info() -> dict:
    import torch
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info |= {"gpu": torch.cuda.get_device_name(0), "gpu_mem_gb": round(p.total_memory / 2**30, 1),
                 "compute_capability": f"{p.major}.{p.minor}", "cuda_runtime": torch.version.cuda}
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import tensorrt as trt
        info["tensorrt"] = trt.__version__
    except ImportError:
        pass
    return info


def timeit(fn, sync, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def feats_for(spec, f: dict, n: int, int_dtype=np.int64) -> dict[str, np.ndarray]:
    reps = int(np.ceil(n / len(f["state"])))
    out = {}
    for k, dt, _ in spec:
        a = np.concatenate([f[k]] * reps)[:n] if reps > 1 else f[k][:n]
        out[k] = np.ascontiguousarray(a.astype(int_dtype if dt == "int64" else np.float32))
    return out


# ---- backends: each returns callable(np-feature-dict) -> np.ndarray [B, D] -------------
def torch_backend(model, spec, device: str, fp16: bool = False):
    import torch

    tower = (model.user if spec is USER_INPUTS else model.item).to(device).eval()

    def run(x):
        t = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in x.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            return tower(t).float().cpu().numpy()
    return run


def ort_backend(path: Path, provider: str, cache_dir: Path, shapes: dict | None = None):
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()          # use the CUDA/cuDNN libs that ship with PyTorch's wheels
        except Exception:  # noqa: BLE001
            pass
    if provider == "cpu":
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return lambda x: sess.run(["embedding"], x)[0]
    if provider == "tensorrt":
        cache_dir.mkdir(parents=True, exist_ok=True)
        opts = {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir), "trt_max_workspace_size": 2 << 30}
        if shapes:
            opts |= {"trt_profile_min_shapes": shapes["min"], "trt_profile_opt_shapes": shapes["opt"],
                     "trt_profile_max_shapes": shapes["max"]}
        providers = [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=providers)
    want = "TensorrtExecutionProvider" if provider == "tensorrt" else "CUDAExecutionProvider"
    if want not in sess.get_providers():
        raise RuntimeError(f"{want} not active (got {sess.get_providers()})")

    def run(x):
        return sess.run(["embedding"], x)[0]
    return run


def shape_strings(spec, sizes: dict, bmin: int, bopt: int, bmax: int) -> dict:
    def one(b):
        parts = []
        for n, _, dims in spec:
            d = [sizes.get(x, x) if isinstance(x, str) else x for x in dims]
            parts.append(f"{n}:" + "x".join(str(v) for v in [b, *d]))
        return ",".join(parts)
    return {"min": one(bmin), "opt": one(bopt), "max": one(bmax)}


class TrtEngine:
    """Native TensorRT: build from ONNX with the Python API, run with torch device buffers."""

    def __init__(self, repo: Path, spec, sizes: dict, bmin: int, bopt: int, bmax: int, plan_out: Path):
        import tensorrt as trt
        import torch

        self.trt, self.torch = trt, torch
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        major = int(trt.__version__.split(".")[0])
        flags = 0 if major >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        net = builder.create_network(flags)
        # TensorRT <= 10: FP32 graph + FP16 builder flag (weakly typed).
        # TensorRT >= 11: always strongly typed, no FP16 flag -> build from the FP16 graph.
        has_flag = hasattr(trt.BuilderFlag, "FP16")
        onnx_path = repo / "user_tower_trt" / ("model_int32.onnx" if has_flag else "model_int32_fp16.onnx")
        self.precision_mode = "fp16 via builder flag" if has_flag else "fp16 via strongly typed fp16 graph"
        parser = trt.OnnxParser(net, logger)
        if not parser.parse(onnx_path.read_bytes()):
            n_err = getattr(parser, "num_errors", 1)
            raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(n_err)))
        cfgb = builder.create_builder_config()
        cfgb.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        if has_flag:
            cfgb.set_flag(trt.BuilderFlag.FP16)
        prof = builder.create_optimization_profile()
        self.extra = {}
        for n, _, dims in spec:
            d = [sizes.get(x, x) if isinstance(x, str) else x for x in dims]
            prof.set_shape(n, (bmin, *d), (bopt, *d), (bmax, *d))
            self.extra[n] = d
        cfgb.add_optimization_profile(prof)
        t0 = time.perf_counter()
        plan = builder.build_serialized_network(net, cfgb)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")
        self.build_seconds = time.perf_counter() - t0
        plan_out.parent.mkdir(parents=True, exist_ok=True)
        plan_out.write_bytes(bytes(plan))
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]

    def __call__(self, x: dict[str, np.ndarray]) -> np.ndarray:
        torch = self.torch
        B = len(next(iter(x.values())))
        keep = []
        for n in self.inputs:
            t = torch.from_numpy(x[n]).cuda(non_blocking=True)
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
            keep.append(t)
        out = torch.empty((B, *self.engine.get_tensor_shape(self.outputs[0])[1:]), dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.outputs[0], out.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return out.cpu().numpy()


def parity(ref: np.ndarray, got: np.ndarray) -> dict:
    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-12)
    return {"max_abs_diff": float(np.abs(ref - got).max()), "min_cosine": float(cos.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/colab_a100.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--batches", default="1,8,64,256,1024,4096")
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    import torch

    from pathrank.features.pipeline import AsOf, load_store
    from pathrank.recommender import Recommender

    d = cfg.art_dir
    repo = d / "model_repository"
    if not (repo / "user_tower" / "1" / "model.onnx").exists():
        raise SystemExit("run the pipeline through the export stage first")
    store = load_store(cfg)
    c = store.cutoffs
    T = AsOf(store, c["test"], c["end"] + pd.Timedelta(seconds=1), cfg.two_tower.hash_buckets)
    rec = Recommender.load(store, cfg, d, backend="torch")
    users = T.eval_users(4096, cfg.seed)
    uf = T.user_features(users)
    jf = T.job_features(T.catalog[:4096])
    sizes = {"HIST": uf["tok"].shape[1], "TITLE": jf["tok"].shape[1], "TEXT": uf["text"].shape[1]}
    batches = [int(b) for b in a.batches.split(",")]
    bmax = max(batches)
    gpu = torch.cuda.is_available()
    sync = torch.cuda.synchronize if gpu else (lambda: None)

    res: dict = {"env": env_info(), "batches": batches, "iters": a.iters, "user_tower": {}, "item_tower": {},
                 "skipped": {}}
    log.info("environment: %s", res["env"])

    for tower_name, spec, f, onnx_name in (("user_tower", USER_INPUTS, uf, "user_tower"),
                                           ("item_tower", ITEM_INPUTS, jf, "item_tower")):
        ref_in = feats_for(spec, f, 1024)
        ref = torch_backend(rec.two_tower, spec, "cpu")(ref_in)
        backends: dict = {"torch_cpu": (torch_backend(rec.two_tower, spec, "cpu"), np.int64)}
        if gpu:
            backends["torch_cuda"] = (torch_backend(rec.two_tower, spec, "cuda"), np.int64)
            backends["torch_cuda_fp16"] = (torch_backend(rec.two_tower, spec, "cuda", fp16=True), np.int64)
        onnx = repo / onnx_name / "1" / "model.onnx"
        backends["ort_cpu"] = (ort_backend(onnx, "cpu", d), np.int64)
        for name, prov in (("ort_cuda", "cuda"), ("ort_tensorrt", "tensorrt")):
            try:
                if not gpu:
                    raise RuntimeError("no CUDA device")
                shp = shape_strings(spec, sizes, 1, 256, bmax) if prov == "tensorrt" else None
                backends[name] = (ort_backend(onnx, prov, d / "gpu" / "ort_trt_cache", shp), np.int64)
            except Exception as e:  # noqa: BLE001
                res["skipped"][f"{tower_name}.{name}"] = f"{type(e).__name__}: {str(e)[:200]}"
        if tower_name == "user_tower":
            try:
                if not gpu:
                    raise RuntimeError("no CUDA device")
                i32 = [(n, "int32" if dt == "int64" else dt, sp) for n, dt, sp in USER_INPUTS]
                eng = TrtEngine(repo, i32, sizes, 1, 256, bmax, d / "gpu" / "user_tower_fp16.plan")
                res["tensorrt_engine_build_s"] = round(eng.build_seconds, 1)
                res["tensorrt_precision_mode"] = eng.precision_mode
                backends["tensorrt"] = (eng, np.int32)
            except Exception as e:  # noqa: BLE001
                res["skipped"][f"{tower_name}.tensorrt"] = f"{type(e).__name__}: {str(e)[:200]}"
                log.debug(traceback.format_exc())

        for name, (fn, idt) in backends.items():
            try:
                got = fn(feats_for(spec, f, 1024, idt))
                row = {"parity": parity(ref, got), "latency_ms": {}, "throughput_rows_s": {}}
                for b in batches:
                    if name in ("torch_cpu", "ort_cpu") and b > 1024:
                        continue
                    x = feats_for(spec, f, b, idt)
                    ts = timeit(lambda: fn(x), sync, warmup=5, iters=a.iters if b <= 1024 else max(10, a.iters // 5))
                    row["latency_ms"][b] = {"p50": round(float(np.percentile(ts, 50)), 4),
                                            "p99": round(float(np.percentile(ts, 99)), 4)}
                    row["throughput_rows_s"][b] = round(b / (np.median(ts) / 1000))
                res[tower_name][name] = row
                log.info("%s %-16s parity %s  b1 p50 %s ms  b%d %s rows/s", tower_name, name, row["parity"],
                         row["latency_ms"].get(1, {}).get("p50"), max(row["throughput_rows_s"]),
                         f"{row['throughput_rows_s'][max(row['throughput_rows_s'])]:,}")
            except Exception as e:  # noqa: BLE001
                res["skipped"][f"{tower_name}.{name}"] = f"{type(e).__name__}: {str(e)[:200]}"

    # speedups vs PyTorch CPU and vs PyTorch CUDA at the largest common batch and at batch 1
    for tower in ("user_tower", "item_tower"):
        t = res[tower]
        base_cpu = t.get("torch_cpu")
        base_gpu = t.get("torch_cuda")
        for name, row in t.items():
            if base_cpu:
                row["speedup_vs_torch_cpu_b1"] = round(base_cpu["latency_ms"][1]["p50"] / row["latency_ms"][1]["p50"], 2)
            if base_gpu and 1024 in row["throughput_rows_s"]:
                row["throughput_vs_torch_cuda_b1024"] = round(row["throughput_rows_s"][1024]
                                                              / base_gpu["throughput_rows_s"][1024], 2)
    save_json(res, d / "gpu_benchmark.json")
    rp = d / "results.json"
    allr = load_json(rp) if rp.exists() else {}
    allr["gpu"] = res
    save_json(allr, rp)
    print(json.dumps({"env": res["env"], "skipped": res["skipped"]}, indent=2))


if __name__ == "__main__":
    main()
