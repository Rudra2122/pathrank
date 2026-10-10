"""Text encoders for job descriptions and user profiles.

backend=st     sentence-transformers (all-MiniLM-L6-v2, 384-d). Default on a laptop with internet.
backend=tfidf  TF-IDF + TruncatedSVD (128-d). Offline fallback; also what CI uses.
backend=auto   try st, fall back to tfidf with a warning.
backend=none   zero vectors, for data with no text (XING public copy).
The fitted encoder is pickled so serving encodes new users exactly as training did.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..utils import log


class TextEncoder:
    def __init__(self, backend: str, st_model: str = "", dim: int = 128, seed: int = 0):
        self.backend, self.st_model, self.dim, self.seed = backend, st_model, dim, seed
        self._st = None
        self._tfidf = None
        self._svd = None

    # -- fit ----------------------------------------------------------------
    def fit(self, texts: list[str], fit_rows: int = 200_000) -> "TextEncoder":
        if self.backend == "none":
            self.dim = min(self.dim, 16)
            return self
        if self.backend in ("auto", "st"):
            try:
                self._load_st()
                self.backend = "st"
                self.dim = self._st.get_sentence_embedding_dimension()
                return self
            except Exception as e:  # noqa: BLE001
                if self.backend == "st":
                    raise
                log.warning("sentence-transformers unavailable (%s); using tfidf+svd", type(e).__name__)
                self.backend = "tfidf"
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(texts), min(fit_rows, len(texts)), replace=False)
        sample = [texts[i] for i in idx]
        self._tfidf = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), min_df=3,
                                      sublinear_tf=True, dtype=np.float32)
        X = self._tfidf.fit_transform(sample)
        self.dim = min(self.dim, X.shape[1] - 1)
        self._svd = TruncatedSVD(self.dim, random_state=self.seed).fit(X)
        return self

    def _load_st(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            self._st = SentenceTransformer(self.st_model, device=dev)

    # -- encode -------------------------------------------------------------
    def encode(self, texts: list[str], batch: int = 256) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "none":
            return np.zeros((len(texts), self.dim), dtype=np.float32)
        if self.backend == "st":
            self._load_st()
            v = self._st.encode(list(texts), batch_size=batch, show_progress_bar=len(texts) > 50_000,
                                normalize_embeddings=True, convert_to_numpy=True)
            return v.astype(np.float32)
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        step = 100_000
        for s in range(0, len(texts), step):
            X = self._tfidf.transform(texts[s:s + step])
            v = self._svd.transform(X).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
            out[s:s + step] = v
        return out

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        st, self._st = self._st, None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._st = st

    @staticmethod
    def load(path: str | Path) -> "TextEncoder":
        with open(path, "rb") as f:
            return pickle.load(f)


def job_text(jobs) -> list[str]:
    return (jobs["title"].fillna("") + ". " + jobs["title"].fillna("") + ". "
            + jobs["description"].fillna("") + " " + jobs["requirements"].fillna("")).tolist()


def user_text(users, hist_titles) -> list[str]:
    """Profile text: degree + major + past titles (work history and applied-job titles)."""
    return (users["degree"].fillna("").astype(str) + " " + users["major"].fillna("").astype(str) + ". "
            + users["work_titles"].fillna("").astype(str) + " | " + hist_titles.fillna("").astype(str)).tolist()
