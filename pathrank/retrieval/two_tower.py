"""Two-tower retrieval in PyTorch.

Both towers are content-based (location, degree, major, hashed title tokens, text embedding,
activity), not ID-based. That matters on this data: most jobs in a test window were posted
after training ended, so an ID-embedding model cannot score them at all.

Training: in-batch sampled softmax with temperature and logQ popularity correction
(Yi et al., 2019). Users are featurized as-of `ctx`; targets are applications in [ctx, val).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features.pipeline import AsOf
from ..utils import log


def _bag(emb: nn.Embedding, tok: torch.Tensor) -> torch.Tensor:
    m = (tok > 0).to(emb.weight.dtype).unsqueeze(-1)    # follows weight dtype (fp32 / fp16 export)
    return (emb(tok) * m).sum(1) / m.sum(1).clamp(min=1.0)


class UserTower(nn.Module):
    def __init__(self, text_dim, buckets, d=32, hidden=256, out=64):
        super().__init__()
        self.state = nn.Embedding(256, d)
        self.city = nn.Embedding(4096 * 8, d)
        self.degree = nn.Embedding(8, d)
        self.major = nn.Embedding(4096, d)
        self.tok = nn.Embedding(buckets, d, padding_idx=0)
        self.text = nn.Linear(text_dim, d)
        self.num = nn.Linear(6, d)
        self.mlp = nn.Sequential(nn.Linear(7 * d, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, b):
        x = torch.cat([self.state(b["state"]), self.city(b["city"]), self.degree(b["degree"]),
                       self.major(b["major"]), _bag(self.tok, b["tok"]), self.text(b["text"]),
                       self.num(b["num"])], -1)
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, text_dim, buckets, d=32, hidden=256, out=64):
        super().__init__()
        self.state = nn.Embedding(256, d)
        self.city = nn.Embedding(4096 * 8, d)
        self.tok = nn.Embedding(buckets, d, padding_idx=0)
        self.text = nn.Linear(text_dim, d)
        self.mlp = nn.Sequential(nn.Linear(4 * d, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, b):
        x = torch.cat([self.state(b["state"]), self.city(b["city"]), _bag(self.tok, b["tok"]),
                       self.text(b["text"])], -1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, text_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.user = UserTower(text_dim, cfg.hash_buckets, hidden=cfg.hidden, out=cfg.dim)
        self.item = ItemTower(text_dim, cfg.hash_buckets, hidden=cfg.hidden, out=cfg.dim)
        # torch's default N(0,1) embedding init swamps the dense inputs; small init trains
        # much better here (found by comparing against the Keras baseline, which uses +-0.05)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.05)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()


def _to_t(feats: dict, idx: np.ndarray, device: torch.device | None = None) -> dict:
    out = {}
    for k, v in feats.items():
        if k in ("uidx", "jidx"):
            continue
        a = v[idx]
        t = torch.from_numpy(a.astype(np.int64) if a.dtype.kind in "iu" else a.astype(np.float32))
        out[k] = t.to(device, non_blocking=True) if device is not None and device.type != "cpu" else t
    return out


def _tensors(feats: dict, device: torch.device) -> dict:
    """All training features as tensors on the training device, once (no per-batch host copies)."""
    out = {}
    for k, v in feats.items():
        if k in ("uidx", "jidx"):
            continue
        out[k] = torch.from_numpy(v.astype(np.int64) if v.dtype.kind in "iu" else v.astype(np.float32)).to(device)
    return out


def _take(tf: dict, idx: torch.Tensor) -> dict:
    return {k: v.index_select(0, idx) for k, v in tf.items()}


def train_two_tower(ctx: AsOf, cfg, seed: int = 0, tracker=None) -> TwoTower:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = ctx.labels[["uidx", "jidx"]].to_numpy()
    if cfg.max_train_pairs and len(pairs) > cfg.max_train_pairs:
        pairs = pairs[rng.choice(len(pairs), cfg.max_train_pairs, replace=False)]
    users = np.unique(pairs[:, 0])
    items = np.unique(pairs[:, 1])
    log.info("two-tower: %s training pairs, %s users, %s items", f"{len(pairs):,}", f"{len(users):,}",
             f"{len(items):,}")
    uf = ctx.user_features(users)
    jf = ctx.job_features(items)
    upos = np.searchsorted(users, pairs[:, 0])
    ipos = np.searchsorted(items, pairs[:, 1])
    item_freq = np.bincount(ipos, minlength=len(items)).astype(np.float64)

    from ..device import describe, resolve, sync
    dev = resolve(getattr(cfg, "device", "auto"))
    log.info("two-tower training on %s", describe(dev))
    logq = torch.from_numpy(np.log(item_freq / item_freq.sum()).astype(np.float32)).to(dev)
    ut, it = _tensors(uf, dev), _tensors(jf, dev)
    upos_t = torch.from_numpy(upos).to(dev)
    ipos_t = torch.from_numpy(ipos).to(dev)

    model = TwoTower(uf["text"].shape[1], cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    B = cfg.batch_size
    import time
    t0 = time.perf_counter()
    for ep in range(cfg.epochs):
        perm = torch.from_numpy(rng.permutation(len(pairs))).to(dev)
        tot = torch.zeros((), device=dev)
        n = 0
        model.train()
        for s in range(0, len(perm), B):
            bi = perm[s:s + B]
            ib_pos = ipos_t[bi]
            u = model.user(_take(ut, upos_t[bi]))
            v = model.item(_take(it, ib_pos))
            logits = u @ v.T / cfg.temperature
            if cfg.logq_correction:
                logits = logits - logq[ib_pos].unsqueeze(0)
            # mask in-batch duplicates of the positive item (false negatives)
            same = ib_pos[:, None] == ib_pos[None, :]
            eye = torch.eye(len(bi), dtype=torch.bool, device=dev)
            logits = logits.masked_fill(same & ~eye, -1e9)
            loss = F.cross_entropy(logits, torch.arange(len(bi), device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach() * len(bi)          # no per-step host sync
            n += len(bi)
        ep_loss = float(tot) / n
        log.info("two-tower epoch %d loss %.4f", ep + 1, ep_loss)
        if tracker:
            tracker.metrics({"two_tower_loss": ep_loss}, step=ep)
    sync(dev)
    model.train_seconds = time.perf_counter() - t0
    model.train_device = describe(dev)
    model.pairs_per_s = cfg.epochs * len(pairs) / max(model.train_seconds, 1e-9)
    model.eval()
    return model.cpu()        # export, serving and ONNX all expect a CPU module


def _dev(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def encode_users(model: TwoTower, uf: dict, batch: int = 8192) -> np.ndarray:
    n = len(uf["uidx"])
    d = _dev(model)
    out = np.empty((n, model.cfg.dim), np.float32)
    for s in range(0, n, batch):
        out[s:s + batch] = model.user(_to_t(uf, np.arange(s, min(s + batch, n)), d)).cpu().numpy()
    return out


@torch.no_grad()
def encode_items(model: TwoTower, jf: dict, batch: int = 16384) -> np.ndarray:
    n = len(jf["jidx"])
    d = _dev(model)
    out = np.empty((n, model.cfg.dim), np.float32)
    for s in range(0, n, batch):
        out[s:s + batch] = model.item(_to_t(jf, np.arange(s, min(s + batch, n)), d)).cpu().numpy()
    return out
