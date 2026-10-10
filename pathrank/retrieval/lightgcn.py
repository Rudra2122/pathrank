"""LightGCN (He et al., 2020) on the user-job bipartite graph, in plain PyTorch sparse ops.

(Plain torch instead of PyTorch Geometric: same math, no compiled-extension install pain on
Apple Silicon. A PyG version is ~10 lines of LGConv if you want the keyword.)

It is ID-based, so it can only score jobs that already have applications in the graph.
On job data with heavy posting churn that limits it to warm jobs; that is a finding worth
reporting, not hiding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..utils import log

import warnings
warnings.filterwarnings("ignore", message=".*[Ss]parse.*")


class _SymSpMM(torch.autograd.Function):
    """y = A @ x for a symmetric sparse A. Backward is A @ grad (A^T = A), so no autograd
    through sparse ops: deterministic, and the same CSR kernel on CPU and CUDA."""

    @staticmethod
    def forward(ctx, A, x):
        ctx.A = A
        return A @ x

    @staticmethod
    def backward(ctx, grad):
        return None, ctx.A @ grad


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim, layers):
        super().__init__()
        self.n_users, self.n_items, self.layers = n_users, n_items, layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.A = None

    def set_graph(self, u: np.ndarray, i: np.ndarray):
        n = self.n_users + self.n_items
        rows = np.concatenate([u, i + self.n_users])
        cols = np.concatenate([i + self.n_users, u])
        deg = np.bincount(rows, minlength=n).astype(np.float32)
        dinv = np.power(np.maximum(deg, 1), -0.5)
        vals = (dinv[rows] * dinv[cols]).astype(np.float32)
        # build CSR directly (scipy) instead of COO->coalesce->CSR: ~half the peak memory
        import scipy.sparse as sp
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
        A.sum_duplicates()
        self.A = torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)),
                                         torch.from_numpy(A.indices.astype(np.int64)),
                                         torch.from_numpy(A.data), (n, n))

    def to_device(self, dev: torch.device) -> "LightGCN":
        self.to(dev)
        self.A = self.A.to(dev)
        return self

    def propagate(self):
        x = self.emb.weight
        acc = x
        for _ in range(self.layers):
            x = _SymSpMM.apply(self.A, x)
            acc = acc + x
        out = acc / (self.layers + 1)
        return out[: self.n_users], out[self.n_users:]


class GCNRetriever:
    """Holds trained embeddings + id maps; scores catalog jobs that exist in the graph."""

    def __init__(self, user_ids, item_ids, U, I):
        self.user_pos = pd.Series(np.arange(len(user_ids)), index=user_ids)
        self.item_ids = item_ids
        self.item_pos = pd.Series(np.arange(len(item_ids)), index=item_ids)
        self.U, self.I = U, I

    def retrieve(self, uidx: np.ndarray, catalog: np.ndarray, k: int):
        from .index import CatalogIndex

        cat = catalog[np.isin(catalog, self.item_ids)]
        n = len(uidx)
        S = np.full((n, k), -np.inf, np.float32)
        J = np.full((n, k), -1, np.int64)
        known = np.isin(uidx, self.user_pos.index)
        if len(cat) == 0 or not known.any():
            return S, J, 0
        idx = CatalogIndex(self.I[self.item_pos[cat].to_numpy()], cat)
        q = self.U[self.user_pos[uidx[known]].to_numpy()]
        s, j = idx.search(q, k)
        S[known, : s.shape[1]], J[known, : j.shape[1]] = s, j
        return S, J, len(cat)


def train_lightgcn(edges: pd.DataFrame, cfg, seed: int = 0, tracker=None) -> GCNRetriever:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    edges = edges.drop_duplicates(["uidx", "jidx"])
    user_ids = np.sort(edges.uidx.unique())
    item_ids = np.sort(edges.jidx.unique())
    u = np.searchsorted(user_ids, edges.uidx.to_numpy())
    i = np.searchsorted(item_ids, edges.jidx.to_numpy())
    log.info("lightgcn: %s users, %s items, %s edges", f"{len(user_ids):,}", f"{len(item_ids):,}",
             f"{len(u):,}")
    from ..device import describe, resolve, sync
    dev = resolve(getattr(cfg, "device", "auto"), sparse=True)
    log.info("lightgcn training on %s", describe(dev))
    m = LightGCN(len(user_ids), len(item_ids), cfg.dim, cfg.layers)
    m.set_graph(u, i)
    m.to_device(dev)
    u_t, i_t = torch.from_numpy(u).to(dev), torch.from_numpy(i).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    B = cfg.batch_size
    import time
    t0 = time.perf_counter()
    for ep in range(cfg.epochs):
        perm = torch.from_numpy(rng.permutation(len(u))).to(dev)
        tot = torch.zeros((), device=dev)
        for s in range(0, len(perm), B):
            bi = perm[s:s + B]
            neg = torch.from_numpy(rng.integers(0, len(item_ids), len(bi))).to(dev)
            ub, ib = u_t[bi], i_t[bi]
            Ue, Ie = m.propagate()
            # index_select (backward = index_add) instead of advanced indexing: deterministic
            ue, pe, ne = Ue.index_select(0, ub), Ie.index_select(0, ib), Ie.index_select(0, neg)
            pos = (ue * pe).sum(-1)
            negs = (ue * ne).sum(-1)
            loss = -torch.nn.functional.logsigmoid(pos - negs).mean()
            e0 = m.emb.weight
            reg = cfg.reg * (e0.index_select(0, ub).pow(2).sum() + e0.index_select(0, ib + m.n_users).pow(2).sum()
                             + e0.index_select(0, neg + m.n_users).pow(2).sum()) / len(bi)
            opt.zero_grad()
            (loss + reg).backward()
            opt.step()
            tot += loss.detach() * len(bi)
        ep_loss = float(tot) / len(u)
        log.info("lightgcn epoch %d bpr %.4f", ep + 1, ep_loss)
        if tracker:
            tracker.metrics({"lightgcn_bpr": ep_loss}, step=ep)
    sync(dev)
    train_s = time.perf_counter() - t0
    with torch.no_grad():
        Ue, Ie = m.propagate()
        Ue = torch.nn.functional.normalize(Ue, dim=-1).cpu().numpy()
        Ie = torch.nn.functional.normalize(Ie, dim=-1).cpu().numpy()
    r = GCNRetriever(user_ids, item_ids, Ue, Ie)
    r.train_seconds, r.train_device = train_s, describe(dev)
    return r
