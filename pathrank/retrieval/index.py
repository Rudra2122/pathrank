"""FAISS wrapper restricted to a catalog subset (only open jobs are retrievable)."""
from __future__ import annotations

import numpy as np


class CatalogIndex:
    def __init__(self, item_vecs: np.ndarray, item_ids: np.ndarray, kind: str = "flat"):
        import sys

        import faiss

        if sys.platform == "darwin":
            faiss.omp_set_num_threads(1)     # avoid libomp clash with torch on macOS
        self.ids = np.asarray(item_ids)
        x = np.ascontiguousarray(item_vecs.astype(np.float32))
        d = x.shape[1]
        if kind == "hnsw" and len(x) > 50_000:
            self.index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 256
        else:
            self.index = faiss.IndexFlatIP(d)
        self.index.add(x)

    def search(self, q: np.ndarray, k: int, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.ids))
        S = np.empty((len(q), k), np.float32)
        I = np.empty((len(q), k), np.int64)
        q = np.ascontiguousarray(q.astype(np.float32))
        for s in range(0, len(q), batch):
            sc, ix = self.index.search(q[s:s + batch], k)
            S[s:s + batch], I[s:s + batch] = sc, ix
        valid = I >= 0
        out = np.where(valid, self.ids[np.clip(I, 0, None)], -1)
        return S, out
