"""Test double for Triton Inference Server: speaks the KServe v2 HTTP protocol (JSON header +
binary tensors) and ENFORCES the repository's config.pbtxt files (dtype, shape, max batch,
reshape), executing ONNX models with onnxruntime and the FIL ranker with LightGBM.

It is not Triton. It exists so CI can prove the client, the exported models and the
configs agree with each other without a GPU or the Triton container.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

_NP = {"TYPE_FP32": np.float32, "TYPE_INT64": np.int64, "TYPE_INT32": np.int32}
_WIRE = {"TYPE_FP32": "FP32", "TYPE_INT64": "INT64", "TYPE_INT32": "INT32"}
_FROM_WIRE = {"FP32": np.float32, "INT64": np.int64, "INT32": np.int32}


class Model:
    def __init__(self, d: Path):
        from google.protobuf import text_format
        from tritonclient.grpc import model_config_pb2

        self.cfg = text_format.Parse((d / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
        self.name = self.cfg.name
        f = d / "1"
        if (f / "model.onnx").exists():
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(f / "model.onnx"), providers=["CPUExecutionProvider"])
            self.kind = "onnx"
        elif (f / "model.txt").exists() and self.cfg.backend == "fil":
            import lightgbm as lgb
            self.booster = lgb.Booster(model_file=str(f / "model.txt"))
            self.kind = "fil"
        else:
            raise FileNotFoundError(f"no loadable model in {f}")

    def infer(self, tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feeds = {}
        batch = None
        for spec in self.cfg.input:
            if spec.name not in tensors:
                raise ValueError(f"missing input {spec.name}")
            a = tensors[spec.name]
            want = list(spec.dims)
            if a.dtype != _NP[model_dtype(spec.data_type)]:
                raise ValueError(f"{spec.name}: dtype {a.dtype} != {model_dtype(spec.data_type)}")
            if list(a.shape[1:]) != want:
                raise ValueError(f"{spec.name}: shape {list(a.shape)} != [-1] + {want}")
            batch = a.shape[0] if batch is None else batch
            if a.shape[0] != batch or batch > self.cfg.max_batch_size:
                raise ValueError(f"{spec.name}: bad batch {a.shape[0]} (max {self.cfg.max_batch_size})")
            if spec.HasField("reshape"):
                a = a.reshape([batch] + list(spec.reshape.shape))
            feeds[spec.name] = a
        if self.kind == "onnx":
            names = [o.name for o in self.cfg.output]
            outs = dict(zip(names, self.sess.run(names, feeds)))
        else:
            x = feeds["input__0"]
            outs = {"output__0": self.booster.predict(x).astype(np.float32).reshape(-1, 1)}
        for spec in self.cfg.output:
            if list(outs[spec.name].shape[1:]) != list(spec.dims):
                raise ValueError(f"output {spec.name} shape {outs[spec.name].shape} != dims {list(spec.dims)}")
        return outs


def model_dtype(enum_value: int) -> str:
    from tritonclient.grpc import model_config_pb2
    return model_config_pb2.DataType.Name(enum_value)


class _Handler(BaseHTTPRequestHandler):
    models: dict = {}
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes = b"", headers: dict | None = None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/")
        if p in ("/v2/health/ready", "/v2/health/live"):
            return self._send(200)
        parts = p.split("/")
        if len(parts) >= 4 and parts[1:3] == ["v2", "models"] and parts[3] in self.models:
            if p.endswith("/ready"):
                return self._send(200)
            m = self.models[parts[3]]
            return self._send(200, json.dumps({"name": m.name, "platform": m.cfg.platform or m.cfg.backend}).encode(),
                              {"Content-Type": "application/json"})
        return self._send(404)

    def do_POST(self):
        parts = self.path.split("?")[0].strip("/").split("/")
        if len(parts) < 4 or parts[-1] != "infer" or parts[2] not in self.models:
            return self._send(404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        hlen = int(self.headers.get("Inference-Header-Content-Length", len(body)))
        req = json.loads(body[:hlen])
        raw, off, tensors = body[hlen:], 0, {}
        try:
            for i in req["inputs"]:
                dt = _FROM_WIRE[i["datatype"]]
                size = (i.get("parameters") or {}).get("binary_data_size")
                if size is not None:
                    tensors[i["name"]] = np.frombuffer(raw[off:off + size], dtype=dt).reshape(i["shape"])
                    off += size
                else:
                    tensors[i["name"]] = np.asarray(i["data"], dtype=dt).reshape(i["shape"])
            outs = self.models[parts[2]].infer(tensors)
        except Exception as e:  # noqa: BLE001
            return self._send(400, json.dumps({"error": str(e)}).encode(), {"Content-Type": "application/json"})
        want = {o["name"] for o in req.get("outputs", [])} or set(outs)
        meta, blobs = [], []
        for name, a in outs.items():
            if name not in want:
                continue
            b = np.ascontiguousarray(a).tobytes()
            meta.append({"name": name, "datatype": "FP32" if a.dtype == np.float32 else "INT64",
                         "shape": list(a.shape), "parameters": {"binary_data_size": len(b)}})
            blobs.append(b)
        head = json.dumps({"model_name": parts[2], "outputs": meta}).encode()
        self._send(200, head + b"".join(blobs), {"Content-Type": "application/octet-stream",
                                                 "Inference-Header-Content-Length": str(len(head))})


def serve(repo: Path, port: int = 0, skip: tuple = ("user_tower_trt",)):
    models = {d.name: Model(d) for d in sorted(Path(repo).iterdir())
              if d.is_dir() and (d / "config.pbtxt").exists() and d.name not in skip}
    handler = type("H", (_Handler,), {"models": models})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"127.0.0.1:{srv.server_address[1]}"
