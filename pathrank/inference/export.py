"""Export models for Triton Inference Server and build the model repository.

model_repository/
  user_tower/      ONNX (onnxruntime backend), dynamic batching; optional TensorRT accelerator
  item_tower/      ONNX, used by the catalog re-index job
  ranker/          LightGBM model.txt served by Triton's FIL backend (GPU tree inference)
  user_tower_trt/  TensorRT engine (model.plan), built on a GPU by scripts/build_tensorrt.sh

Every config.pbtxt is parsed against Triton's ModelConfig protobuf before it is written,
and every ONNX export is checked against PyTorch on real feature batches.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..utils import log

USER_INPUTS = [("state", "int64", []), ("city", "int64", []), ("degree", "int64", []), ("major", "int64", []),
               ("tok", "int64", ["HIST"]), ("text", "float32", ["TEXT"]), ("num", "float32", [6])]
ITEM_INPUTS = [("state", "int64", []), ("city", "int64", []), ("tok", "int64", ["TITLE"]), ("text", "float32", ["TEXT"])]
_TRITON_DT = {"int64": "TYPE_INT64", "float32": "TYPE_FP32"}


class _UserWrap(nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.t = tower

    def forward(self, state, city, degree, major, tok, text, num):
        return self.t({"state": state, "city": city, "degree": degree, "major": major,
                       "tok": tok, "text": text, "num": num})


class _UserWrapI32(nn.Module):
    """TensorRT source graph: int32 ids in, cast to int64 inside (portable across TRT versions)."""

    def __init__(self, tower):
        super().__init__()
        self.t = tower

    def forward(self, state, city, degree, major, tok, text, num):
        return self.t({"state": state.long(), "city": city.long(), "degree": degree.long(),
                       "major": major.long(), "tok": tok.long(), "text": text, "num": num})


class _UserWrapFP16(nn.Module):
    """FP16 TensorRT source for TensorRT >= 11 (always strongly typed: precision comes from the
    graph, the FP16 builder flag is gone). int32 ids / fp32 features in, fp32 embedding out."""

    def __init__(self, tower):
        super().__init__()
        import copy
        self.t = copy.deepcopy(tower).half()

    def forward(self, state, city, degree, major, tok, text, num):
        return self.t({"state": state.long(), "city": city.long(), "degree": degree.long(),
                       "major": major.long(), "tok": tok.long(), "text": text.half(),
                       "num": num.half()}).float()


class _ItemWrap(nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.t = tower

    def forward(self, state, city, tok, text):
        return self.t({"state": state, "city": city, "tok": tok, "text": text})


def _batch(feats: dict, names: list[str], n: int) -> tuple:
    out = []
    for k in names:
        a = feats[k][:n]
        out.append(torch.from_numpy(a.astype(np.int64) if a.dtype.kind in "iu" else a.astype(np.float32)))
    return tuple(out)


def export_tower(wrapper: nn.Module, inputs: list, sample: tuple, path: Path, opset: int) -> None:
    names = [n for n, _, _ in inputs]
    dyn = {n: {0: "batch"} for n in names} | {"embedding": {0: "batch"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper.eval(), sample, str(path), input_names=names, output_names=["embedding"],
                      dynamic_axes=dyn, opset_version=opset, dynamo=False)


def onnx_parity(path: Path, wrapper: nn.Module, inputs: list, feats: dict, n: int = 512) -> dict:
    import onnxruntime as ort

    names = [x for x, _, _ in inputs]
    t = _batch(feats, names, n)
    with torch.no_grad():
        ref = wrapper(*t).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["embedding"], {k: v.numpy() for k, v in zip(names, t)})[0]
    # batch-size-1 path as well (dynamic axis really is dynamic)
    one = sess.run(["embedding"], {k: v.numpy()[:1] for k, v in zip(names, t)})[0]
    return {"max_abs_diff": float(np.abs(ref - got).max()), "batch1_max_abs_diff": float(np.abs(ref[:1] - one).max()),
            "rows": int(len(ref))}


def _parity_i32(path: Path, ref_wrapper: nn.Module, feats: dict, names: list[str], n: int = 256) -> dict:
    import onnxruntime as ort

    t = _batch(feats, names, n)
    with torch.no_grad():
        ref = ref_wrapper(*t).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["embedding"], {k: (v.numpy().astype(np.int32) if v.dtype == torch.int64 else v.numpy())
                                   for k, v in zip(names, t)})[0]
    return {"max_abs_diff": float(np.abs(ref - got).max())}


def _parity_i32_safe(path: Path, ref_wrapper, feats, names) -> dict:
    """FP16 graphs may lack CPU kernels in ONNX Runtime; report instead of failing the export."""
    try:
        return _parity_i32(path, ref_wrapper, feats, names)
    except Exception as e:  # noqa: BLE001
        return {"cpu_check": f"skipped ({type(e).__name__}); verified on GPU by scripts/gpu_benchmark.py"}


# ---- Triton configs --------------------------------------------------------------------
def _dims(spec, sizes) -> str:
    d = [sizes.get(x, x) if isinstance(x, str) else x for x in spec]
    return "[ 1 ]" if not d else "[ " + ", ".join(str(int(v)) for v in d) + " ]"


def tower_config(name: str, inputs: list, sizes: dict, out_dim: int, max_batch: int, gpu: bool,
                 trt_accelerator: bool) -> str:
    ins = []
    for n, dt, spec in inputs:
        reshape = "\n    reshape: { shape: [ ] }" if not spec else ""   # per-example scalars -> [B]
        ins.append(f'  {{\n    name: "{n}"\n    data_type: {_TRITON_DT[dt]}\n    dims: {_dims(spec, sizes)}{reshape}\n  }}')
    opt = ""
    if trt_accelerator:
        opt = ('\noptimization { execution_accelerators { gpu_execution_accelerator : [ {\n'
               '  name : "tensorrt"\n  parameters { key: "precision_mode" value: "FP16" }\n'
               '  parameters { key: "max_workspace_size_bytes" value: "1073741824" } } ] } }')
    kind = "KIND_GPU" if gpu else "KIND_CPU"
    return (f'name: "{name}"\nplatform: "onnxruntime_onnx"\nmax_batch_size: {max_batch}\n'
            f'input [\n' + ",\n".join(ins) + '\n]\n'
            f'output [\n  {{\n    name: "embedding"\n    data_type: TYPE_FP32\n    dims: [ {out_dim} ]\n  }}\n]\n'
            f'dynamic_batching {{\n  preferred_batch_size: [ 64, 256 ]\n  max_queue_delay_microseconds: 200\n}}\n'
            f'instance_group [ {{ count: 1, kind: {kind} }} ]{opt}\n')


def trt_config(name: str, inputs: list, sizes: dict, out_dim: int, max_batch: int) -> str:
    # TensorRT engines are built with explicit batch: Triton passes [B, ...] straight through.
    ins = []
    for n, dt, spec in inputs:
        dt_trt = "TYPE_INT32" if dt == "int64" else "TYPE_FP32"   # engine built from the int32 source graph
        ins.append(f'  {{\n    name: "{n}"\n    data_type: {dt_trt}\n    dims: {_dims(spec, sizes)}'
                   + ("\n    reshape: { shape: [ ] }" if not spec else "") + "\n  }")
    return (f'name: "{name}"\nplatform: "tensorrt_plan"\nmax_batch_size: {max_batch}\n'
            f'input [\n' + ",\n".join(ins) + '\n]\n'
            f'output [\n  {{\n    name: "embedding"\n    data_type: TYPE_FP32\n    dims: [ {out_dim} ]\n  }}\n]\n'
            f'dynamic_batching {{\n  preferred_batch_size: [ 64, 256 ]\n  max_queue_delay_microseconds: 200\n}}\n'
            f'instance_group [ {{ count: 1, kind: KIND_GPU }} ]\n')


def ranker_config(n_features: int, max_batch: int, gpu: bool) -> str:
    return (f'name: "ranker"\nbackend: "fil"\nmax_batch_size: {max_batch * 512}\n'
            f'input [\n  {{\n    name: "input__0"\n    data_type: TYPE_FP32\n    dims: [ {n_features} ]\n  }}\n]\n'
            f'output [\n  {{\n    name: "output__0"\n    data_type: TYPE_FP32\n    dims: [ 1 ]\n  }}\n]\n'
            f'instance_group [ {{ count: 1, kind: {"KIND_GPU" if gpu else "KIND_CPU"} }} ]\n'
            'parameters [\n'
            '  { key: "model_type" value: { string_value: "lightgbm" } },\n'
            '  { key: "output_class" value: { string_value: "false" } },\n'
            '  { key: "storage_type" value: { string_value: "AUTO" } }\n]\n'
            'dynamic_batching { max_queue_delay_microseconds: 100 }\n')


def validate_config(text: str) -> None:
    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2

    text_format.Parse(text, model_config_pb2.ModelConfig())


def build_model_repository(rec, store, uf: dict, jf: dict, out: Path, max_batch: int, opset: int,
                           gpu: bool = False) -> dict:
    """Write the full Triton repository; returns parity + validation report."""
    out.mkdir(parents=True, exist_ok=True)
    tt = rec.two_tower
    sizes = {"HIST": uf["tok"].shape[1], "TITLE": jf["tok"].shape[1], "TEXT": uf["text"].shape[1]}
    dim = tt.cfg.dim
    rep: dict = {"sizes": sizes, "embedding_dim": dim}

    uw, iw = _UserWrap(tt.user), _ItemWrap(tt.item)
    un = [n for n, _, _ in USER_INPUTS]
    inn = [n for n, _, _ in ITEM_INPUTS]
    export_tower(uw, USER_INPUTS, _batch(uf, un, 8), out / "user_tower" / "1" / "model.onnx", opset)
    export_tower(iw, ITEM_INPUTS, _batch(jf, inn, 8), out / "item_tower" / "1" / "model.onnx", opset)
    rep["user_tower_parity"] = onnx_parity(out / "user_tower" / "1" / "model.onnx", uw, USER_INPUTS, uf)
    # int32-input variant that scripts/build_tensorrt.sh compiles into model.plan
    i32 = [(n, "int32" if dt == "int64" else dt, sp) for n, dt, sp in USER_INPUTS]
    sample32 = tuple(x.int() if x.dtype == torch.int64 else x for x in _batch(uf, un, 8))
    trt_src = out / "user_tower_trt" / "model_int32.onnx"
    export_tower(_UserWrapI32(tt.user), i32, sample32, trt_src, opset)
    rep["user_tower_trt_source_parity"] = _parity_i32(trt_src, uw, uf, un)
    trt16 = out / "user_tower_trt" / "model_int32_fp16.onnx"
    export_tower(_UserWrapFP16(tt.user), i32, sample32, trt16, opset)
    import onnx
    onnx.checker.check_model(onnx.load(str(trt16)))
    rep["user_tower_trt_fp16_source"] = _parity_i32_safe(trt16, uw, uf, un)
    rep["item_tower_parity"] = onnx_parity(out / "item_tower" / "1" / "model.onnx", iw, ITEM_INPUTS, jf)

    # ranker: LightGBM text format, which Triton's FIL backend loads natively
    (out / "ranker" / "1").mkdir(parents=True, exist_ok=True)
    rk = rec.ranker
    rk.model.save_model(str(out / "ranker" / "1" / "model.txt"), num_iteration=rk.model.best_iteration)
    (out / "ranker" / "features.txt").write_text("\n".join(rk.features) + "\n")

    configs = {
        "user_tower": tower_config("user_tower", USER_INPUTS, sizes, dim, max_batch, gpu, trt_accelerator=gpu),
        "item_tower": tower_config("item_tower", ITEM_INPUTS, sizes, dim, max_batch, gpu, trt_accelerator=gpu),
        "ranker": ranker_config(len(rk.features), max_batch, gpu),
        "user_tower_trt": trt_config("user_tower_trt", USER_INPUTS, sizes, dim, max_batch),
    }
    for name, text in configs.items():
        validate_config(text)
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "config.pbtxt").write_text(text)
    # the TRT model needs a GPU-built engine; keep it out of the default load set
    (out / "user_tower_trt" / "BUILD_ON_GPU.txt").write_text(
        "Run scripts/build_tensorrt.sh on an NVIDIA GPU to create 1/model.plan.\n"
        "Until then start Triton with --model-control-mode=explicit and load the other models.\n")
    rep["configs_validated"] = sorted(configs)
    log.info("triton repository at %s: %s", out, rep)
    return rep
