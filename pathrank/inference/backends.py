"""Interchangeable inference backends for the towers and the ranker.

  torch   in-process PyTorch (training default)
  onnx    in-process ONNX Runtime on the exported models (CPU, or CoreML/CUDA providers)
  triton  remote Triton Inference Server over HTTP (KServe v2 protocol, binary tensors)

All three take the same feature dicts / DataFrames and return the same arrays, so the
Recommender doesn't care which one is plugged in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .export import ITEM_INPUTS, USER_INPUTS


def _cast(a: np.ndarray, dt: str) -> np.ndarray:
    return np.ascontiguousarray(a.astype(np.int64 if dt == "int64" else np.float32))


class TorchTowers:
    name = "torch"

    def __init__(self, model):
        self.m = model

    def encode_users(self, uf: dict) -> np.ndarray:
        from ..retrieval.two_tower import encode_users
        return encode_users(self.m, uf)

    def encode_items(self, jf: dict) -> np.ndarray:
        from ..retrieval.two_tower import encode_items
        return encode_items(self.m, jf)


class OnnxTowers:
    name = "onnx"

    def __init__(self, repo: Path, providers: list[str] | None = None):
        import onnxruntime as ort

        avail = ort.get_available_providers()
        prov = providers or [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                                         "CPUExecutionProvider") if p in avail]
        self.u = ort.InferenceSession(str(repo / "user_tower" / "1" / "model.onnx"), providers=prov)
        self.i = ort.InferenceSession(str(repo / "item_tower" / "1" / "model.onnx"), providers=prov)

    def _run(self, sess, spec, feats, batch=8192):
        n = len(next(iter(feats.values())))
        out = []
        for s in range(0, n, batch):
            out.append(sess.run(["embedding"], {k: _cast(feats[k][s:s + batch], dt) for k, dt, _ in spec})[0])
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def encode_users(self, uf):
        return self._run(self.u, USER_INPUTS, uf)

    def encode_items(self, jf):
        return self._run(self.i, ITEM_INPUTS, jf)


class TritonTowers:
    name = "triton"

    def __init__(self, url: str, max_batch: int = 1024, user_model: str = "user_tower"):
        import tritonclient.http as th

        self.th = th
        self.c = th.InferenceServerClient(url=url, connection_timeout=10, network_timeout=30)
        self.max_batch, self.user_model = max_batch, user_model
        if not self.c.is_server_ready():
            raise ConnectionError(f"Triton at {url} not ready")

    def _infer(self, model, spec, feats):
        th = self.th
        n = len(next(iter(feats.values())))
        out = []
        for s in range(0, n, self.max_batch):
            ins = []
            for k, dt, dims in spec:
                a = _cast(feats[k][s:s + self.max_batch], dt)
                if not dims:                 # config declares dims [1] + reshape {shape: []}
                    a = a.reshape(-1, 1)
                x = th.InferInput(k, list(a.shape), "INT64" if dt == "int64" else "FP32")
                x.set_data_from_numpy(a, binary_data=True)
                ins.append(x)
            r = self.c.infer(model, ins, outputs=[th.InferRequestedOutput("embedding", binary_data=True)])
            out.append(r.as_numpy("embedding"))
        return np.concatenate(out)

    def encode_users(self, uf):
        return self._infer(self.user_model, USER_INPUTS, uf)

    def encode_items(self, jf):
        return self._infer("item_tower", ITEM_INPUTS, jf)


class TritonRanker:
    """LightGBM ranker served by Triton's FIL backend. Same .predict(df) as LGBRanker."""

    def __init__(self, url: str, features: list[str], max_rows: int = 65536):
        import tritonclient.http as th

        self.th = th
        self.c = th.InferenceServerClient(url=url, connection_timeout=10, network_timeout=30)
        self.features, self.max_rows = list(features), max_rows

    def predict(self, df) -> np.ndarray:
        th = self.th
        X = np.ascontiguousarray(df[self.features].to_numpy(np.float32))
        out = []
        for s in range(0, len(X), self.max_rows):
            x = th.InferInput("input__0", list(X[s:s + self.max_rows].shape), "FP32")
            x.set_data_from_numpy(X[s:s + self.max_rows], binary_data=True)
            r = self.c.infer("ranker", [x], outputs=[th.InferRequestedOutput("output__0", binary_data=True)])
            out.append(r.as_numpy("output__0").reshape(-1))
        return np.concatenate(out)


def make_towers(backend: str, model=None, repo: Path | None = None, url: str = "", max_batch: int = 1024):
    if backend == "torch":
        return TorchTowers(model)
    if backend == "onnx":
        return OnnxTowers(repo)
    if backend == "triton":
        return TritonTowers(url, max_batch)
    raise ValueError(backend)
