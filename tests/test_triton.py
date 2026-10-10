"""Triton path: exported models + config.pbtxt + HTTP client agree (via a protocol-enforcing stub)."""
import numpy as np
import pandas as pd
import pytest



@pytest.fixture(scope="module")
def triton(built):
    from tests.triton_stub import serve
    srv, url = serve(built.art_dir / "model_repository")
    yield url
    srv.shutdown()


def _ctx(cfg):
    from pathrank.features.pipeline import AsOf, load_store
    s = load_store(cfg)
    c = s.cutoffs
    return s, AsOf(s, c["test"], c["end"] + pd.Timedelta(seconds=1), cfg.two_tower.hash_buckets)


def test_configs_parse_against_triton_schema(built):
    from pathrank.inference.export import validate_config
    repo = built.art_dir / "model_repository"
    names = sorted(p.parent.name for p in repo.glob("*/config.pbtxt"))
    assert names == ["item_tower", "ranker", "user_tower", "user_tower_trt"]
    for p in repo.glob("*/config.pbtxt"):
        validate_config(p.read_text())


def test_triton_towers_match_torch(built, triton):
    from pathrank.inference.backends import TorchTowers, TritonTowers
    from pathrank.recommender import Recommender
    s, T = _ctx(built)
    rec = Recommender.load(s, built, built.art_dir, backend="torch")
    users = T.eval_users(300, 0)
    uf = T.user_features(users)
    tri = TritonTowers(triton, max_batch=128)          # forces several requests
    np.testing.assert_allclose(tri.encode_users(uf), TorchTowers(rec.two_tower).encode_users(uf), atol=1e-5)
    jf = T.job_features(T.catalog[:500])
    np.testing.assert_allclose(tri.encode_items(jf), TorchTowers(rec.two_tower).encode_items(jf), atol=1e-5)


def test_triton_fil_ranker_matches_lightgbm(built, triton):
    import pickle
    from pathrank.inference.backends import TritonRanker
    rk = pickle.load(open(built.art_dir / "ranker.pkl", "rb"))
    df = pd.read_parquet(built.art_dir / "cur_features.parquet").head(3000)
    np.testing.assert_allclose(TritonRanker(triton, rk.features, max_rows=1000).predict(df), rk.predict(df),
                               rtol=1e-5, atol=1e-5)


def test_full_recommender_over_triton(built, triton, monkeypatch):
    from pathrank.recommender import Recommender
    s, T = _ctx(built)
    monkeypatch.setenv("TRITON_URL", triton)
    remote = Recommender.load(s, built, built.art_dir)
    assert remote.tower_backend == "triton"
    local = Recommender.load(s, built, built.art_dir, backend="torch")
    users = T.eval_users(50, 1)
    a = local.rank(T, users).groupby("uidx").head(10)
    T.cache.clear()
    b = remote.rank(T, users).groupby("uidx").head(10)
    assert (a.jidx.to_numpy() == b.jidx.to_numpy()).mean() > 0.99


def test_stub_rejects_wrong_shapes(built, triton):
    """Scalar inputs must be sent as [B, 1] (config dims [1] + reshape); [B] is a client bug."""
    import tritonclient.http as th
    from tritonclient.utils import InferenceServerException
    c = th.InferenceServerClient(triton)
    x = th.InferInput("state", [4], "INT64")
    x.set_data_from_numpy(np.zeros(4, np.int64))
    with pytest.raises(InferenceServerException):
        c.infer("user_tower", [x])
