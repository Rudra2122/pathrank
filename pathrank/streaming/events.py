"""Raw OLX events as Parquet: the file the Java producer replays and the Spark/pandas jobs read.

Schema (one row per raw event, NOT deduplicated, sorted by timestamp):
  user int64, item int64, event string, timestamp int64 (unix seconds)

to_parquet()     interactions.csv (Kaggle) -> events.parquet   (65,502,201 rows, ~0.5 GB zstd)
synth_events()   OLX-shaped synthetic events for the smoke test / CI (no Kaggle token needed)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.olx import EVENTS, _find
from ..utils import log

COLS = ["user", "item", "event", "timestamp"]


def to_parquet(raw_dir: str, out: str, row_group: int = 1_000_000) -> Path:
    import pyarrow as pa
    import pyarrow.compute as pac
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq

    src = _find(Path(raw_dir))
    if src is None:
        raise SystemExit(f"interactions.csv not found under {raw_dir}; run `make olx` (data stage) first")
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    types = {"user": pa.int64(), "item": pa.int64(), "event": pa.string(), "timestamp": pa.int64()}
    log.info("reading %s", src)
    t = pcsv.read_csv(src, convert_options=pcsv.ConvertOptions(include_columns=COLS, column_types=types))
    t = t.select(COLS)
    # replay order = event time. Stable sort keeps the file order for same-second events.
    t = t.take(pac.sort_indices(t, sort_keys=[("timestamp", "ascending")]))
    pq.write_table(t, out_p, row_group_size=row_group, compression="zstd")
    log.info("wrote %s: %s events", out_p, f"{t.num_rows:,}")
    return out_p


def synth_events(out: str, n_events: int = 2_000_000, n_users: int = 120_000, n_items: int = 8_000,
                 days: int = 14, seed: int = 0) -> Path:
    """Heavy-tailed users and items, OLX event mix, two weeks starting 2020-02-12 UTC."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    uw = rng.pareto(1.2, n_users) + 1
    iw = rng.pareto(1.0, n_items) + 1
    user = rng.choice(n_users, n_events, p=uw / uw.sum()).astype(np.int64) * 7 + 11   # non-dense ids
    item = rng.choice(n_items, n_events, p=iw / iw.sum()).astype(np.int64) * 13 + 5
    mix = np.array([0.82, 0.04, 0.03, 0.04, 0.015, 0.005, 0.02, 0.03])
    event = np.array(EVENTS, dtype=object)[rng.choice(len(EVENTS), n_events, p=mix / mix.sum())]
    start = 1_581_465_600                                                              # 2020-02-12
    ts = np.sort(start + rng.integers(0, days * 86400, n_events)).astype(np.int64)
    t = pa.table({"user": user, "item": item, "event": pa.array(event, pa.string()), "timestamp": ts})
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, out_p, row_group_size=500_000, compression="zstd")
    log.info("wrote %s synthetic events -> %s", f"{n_events:,}", out_p)
    return out_p
