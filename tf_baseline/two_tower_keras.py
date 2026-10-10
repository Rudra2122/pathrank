"""TensorFlow/Keras two-tower baseline, same inputs and loss as the PyTorch model.

(TensorFlow Recommenders is unmaintained and breaks on Keras 3, so this is plain Keras;
the in-batch softmax is the same objective tfrs.tasks.Retrieval computes.)

  python tf_baseline/two_tower_keras.py --config configs/careerbuilder.yaml
Writes artifacts/<source>/<version>/tf_baseline.json and merges into results.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from pathrank.config import load_config  # noqa: E402
from pathrank.eval.metrics import per_user_metrics, summarize  # noqa: E402
from pathrank.features.pipeline import AsOf, load_store  # noqa: E402
from pathrank.retrieval.index import CatalogIndex  # noqa: E402
from pathrank.utils import load_json, log, save_json, timer  # noqa: E402


class Bag(tf.keras.layers.Layer):
    def __init__(self, n, d):
        super().__init__()
        self.e = tf.keras.layers.Embedding(n, d)

    def call(self, tok):
        m = tf.cast(tok > 0, tf.float32)[..., None]
        return tf.reduce_sum(self.e(tok) * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1.0)


def tower(kind, text_dim, buckets, d=32, hidden=256, out=64):
    inp = {"state": tf.keras.Input((), dtype="int64", name="state"),
           "city": tf.keras.Input((), dtype="int64", name="city"),
           "tok": tf.keras.Input((None,), dtype="int64", name="tok"),
           "text": tf.keras.Input((text_dim,), name="text")}
    parts = [tf.keras.layers.Embedding(256, d)(inp["state"]), tf.keras.layers.Embedding(4096 * 8, d)(inp["city"]),
             Bag(buckets, d)(inp["tok"]), tf.keras.layers.Dense(d)(inp["text"])]
    if kind == "user":
        inp |= {"degree": tf.keras.Input((), dtype="int64", name="degree"),
                "major": tf.keras.Input((), dtype="int64", name="major"),
                "num": tf.keras.Input((6,), name="num")}
        parts += [tf.keras.layers.Embedding(8, d)(inp["degree"]), tf.keras.layers.Embedding(4096, d)(inp["major"]),
                  tf.keras.layers.Dense(d)(inp["num"])]
    x = tf.keras.layers.Concatenate()(parts)
    x = tf.keras.layers.Dense(hidden, activation="relu")(x)
    x = tf.keras.layers.Dense(out)(x)
    x = tf.keras.layers.UnitNormalization()(x)
    return tf.keras.Model(inp, x)


def _sel(f, idx, keys):
    return {k: f[k][idx] for k in keys}


UK = ["state", "city", "tok", "text", "degree", "major", "num"]
IK = ["state", "city", "tok", "text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/careerbuilder.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    tc = cfg.two_tower
    tf.random.set_seed(cfg.seed)
    store = load_store(cfg)
    c = store.cutoffs
    ctx = AsOf(store, c["ctx"], c["val"], tc.hash_buckets)
    pairs = ctx.labels[["uidx", "jidx"]].to_numpy()
    rng = np.random.default_rng(cfg.seed)
    if tc.max_train_pairs and len(pairs) > tc.max_train_pairs:
        pairs = pairs[rng.choice(len(pairs), tc.max_train_pairs, replace=False)]
    users, items = np.unique(pairs[:, 0]), np.unique(pairs[:, 1])
    uf, jf = ctx.user_features(users), ctx.job_features(items)
    up, ip = np.searchsorted(users, pairs[:, 0]), np.searchsorted(items, pairs[:, 1])
    freq = np.bincount(ip, minlength=len(items)).astype(np.float64)
    logq_all = np.log(freq / freq.sum()).astype(np.float32)

    ut, it = tower("user", uf["text"].shape[1], tc.hash_buckets), tower("item", uf["text"].shape[1], tc.hash_buckets)
    opt = tf.keras.optimizers.Adam(tc.lr)

    @tf.function(reduce_retracing=True)
    def step(ub, ib, logq, same):
        with tf.GradientTape() as tape:
            u, v = ut(ub, training=True), it(ib, training=True)
            logits = tf.matmul(u, v, transpose_b=True) / tc.temperature - logq[None, :]
            n = tf.shape(logits)[0]
            mask = tf.logical_and(same, tf.logical_not(tf.eye(n, dtype=tf.bool)))
            logits = tf.where(mask, -1e9, logits)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.range(n), logits))
        vars_ = ut.trainable_variables + it.trainable_variables
        opt.apply_gradients(zip(tape.gradient(loss, vars_), vars_))
        return loss

    res: dict = {}
    with timer("tf_train", res):
        for ep in range(tc.epochs):
            perm = rng.permutation(len(pairs))
            tot = []
            for s in range(0, len(perm), tc.batch_size):
                bi = perm[s:s + tc.batch_size]
                same = ip[bi][:, None] == ip[bi][None, :]
                tot.append(float(step(_sel(uf, up[bi], UK), _sel(jf, ip[bi], IK),
                                      logq_all[ip[bi]] if tc.logq_correction else np.zeros(len(bi), np.float32), same)))
            log.info("tf two-tower epoch %d loss %.4f", ep + 1, np.mean(tot))

    T = AsOf(store, c["test"], c["end"] + pd.Timedelta(seconds=1), tc.hash_buckets)
    ev = T.eval_users(cfg.ranker.eval_users, cfg.seed)
    pos = {u: {j for j in s if T.catalog_mask[j]} for u, s in T.label_sets(ev).items()}
    tf_ = T.user_features(ev)
    iv = it.predict(_sel(T.job_features(T.catalog), slice(None), IK), batch_size=8192, verbose=0)
    uv = ut.predict(_sel(tf_, slice(None), UK), batch_size=8192, verbose=0)
    S, J = CatalogIndex(iv, T.catalog).search(uv, 100)
    df = pd.DataFrame({"uidx": np.repeat(ev, J.shape[1]), "jidx": J.ravel(), "score": S.ravel()})
    res["tf_two_tower"] = summarize(per_user_metrics(df[df.jidx >= 0], pos, ev))
    log.info("TF two-tower test: %s", res["tf_two_tower"])
    save_json(res, cfg.art_dir / "tf_baseline.json")
    rp = cfg.art_dir / "results.json"
    allr = load_json(rp) if rp.exists() else {}
    allr["tf_baseline"] = res
    save_json(allr, rp)


if __name__ == "__main__":
    main()
