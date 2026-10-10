from __future__ import annotations

import json
import logging
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

log = logging.getLogger("pathrank")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


@contextmanager
def timer(name: str, sink: dict | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    log.info("%s: %.1fs", name, dt)
    if sink is not None:
        sink[f"time_{name}_s"] = round(dt, 2)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"and", "or", "the", "a", "an", "of", "for", "to", "in", "with", "on", "at", "i", "ii", "iii"}


def tokenize(text: str | None) -> list[str]:
    if not isinstance(text, str):
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def hash_tokens(texts, buckets: int, max_len: int) -> np.ndarray:
    """Deterministic hashed token ids, padded with 0. Shape [N, max_len]."""
    from sklearn.utils import murmurhash3_32

    out = np.zeros((len(texts), max_len), dtype=np.int32)
    cache: dict[str, int] = {}
    for i, t in enumerate(texts):
        toks = tokenize(t)[-max_len:] if isinstance(t, str) else []
        for j, tok in enumerate(toks):
            h = cache.get(tok)
            if h is None:
                h = 1 + (murmurhash3_32(tok, positive=True) % (buckets - 1))
                cache[tok] = h
            out[i, j] = h
    return out


def hash_cat(values, buckets: int) -> np.ndarray:
    from sklearn.utils import murmurhash3_32

    cache: dict = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).lower()
        h = cache.get(s)
        if h is None:
            h = 0 if s == "" else 1 + murmurhash3_32(s, positive=True) % (buckets - 1)
            cache[s] = h
        out[i] = h
    return out


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
