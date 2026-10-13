"""PathRank streaming path CLI. See streaming/README.md for the full walkthrough.

  to-parquet     OLX interactions.csv -> events.parquet (sorted by time; what the producer replays)
  synth          OLX-shaped synthetic events.parquet (smoke test without Kaggle)
  head           first N events of a Parquet file -> new file (shorter lag runs)
  kafka-files    events.parquet -> Kafka-shaped Parquet chunks (stream job without Kafka, for tests)
  stream         Spark Structured Streaming job: Kafka -> S3 (MinIO)
  batch-parity   Spark batch as-of features vs pandas AsOf, with timings
  stream-parity  streaming windows vs Spark batch windows, Kafka -> S3 completeness
  report         results/streaming/*.json -> results/RESULTS_streaming.md + README table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathrank.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["to-parquet", "synth", "head", "kafka-files", "stream", "batch-parity",
                                    "stream-parity", "report"])
    ap.add_argument("--config", default="configs/olx_streaming.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. streaming.sink=local")
    ap.add_argument("--events", help="events Parquet (default: streaming.events_parquet)")
    ap.add_argument("--out", help="output path (to-parquet / synth / head / kafka-files)")
    ap.add_argument("--n", type=int, default=2_000_000, help="rows for synth / head")
    ap.add_argument("--chunk", type=int, default=200_000, help="rows per file for kafka-files")
    ap.add_argument("--source", choices=["kafka", "files"], default="kafka")
    ap.add_argument("--files-dir", help="Kafka-shaped Parquet dir for --source files")
    ap.add_argument("--idle", type=float, default=30, help="stream: stop after N s without new data")
    ap.add_argument("--expect", type=int, help="stream: stop once this many events landed")
    ap.add_argument("--reset", action="store_true", help="stream: drop checkpoints and lake outputs first")
    ap.add_argument("--producer-rate", type=int, help="stream: replay rate, recorded with the lag numbers")
    ap.add_argument("--reference", choices=["rebuild", "store"], default="rebuild",
                    help="batch-parity: pandas side from events (rebuild) or the built feature store")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    events = a.events or cfg.streaming.events_parquet

    if a.cmd == "to-parquet":
        from pathrank.streaming.events import to_parquet
        to_parquet(str(Path(cfg.data.raw_dir) / "olx"), a.out or events)
    elif a.cmd == "synth":
        from pathrank.streaming.events import synth_events
        synth_events(a.out or events, n_events=a.n, seed=cfg.seed)
    elif a.cmd == "head":
        import pyarrow.parquet as pq
        t = pq.read_table(events).slice(0, a.n)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, a.out, row_group_size=500_000, compression="zstd")
        print(f"wrote {t.num_rows:,} rows -> {a.out}")
    elif a.cmd == "kafka-files":
        _kafka_files(events, a.out, a.chunk, cfg.streaming.partitions)
    elif a.cmd == "stream":
        from pathrank.streaming.stream import run
        rate = a.producer_rate
        lag_json = Path(cfg.streaming.results_dir) / "producer_lag.json"
        if rate is None and lag_json.exists():
            rate = json.loads(lag_json.read_text()).get("rate_limit") or None
        run(cfg, a.source, a.files_dir, a.idle, a.expect, a.reset, producer_rate=rate)
    elif a.cmd == "batch-parity":
        from pathrank.streaming.parity import batch_parity
        r = batch_parity(cfg, events, a.reference)
        sys.exit(0 if r["pass"] else 1)
    elif a.cmd == "stream-parity":
        from pathrank.streaming.parity import stream_parity
        r = stream_parity(cfg, events)
        sys.exit(0 if r["pass"] else 1)
    elif a.cmd == "report":
        from pathrank.streaming.report import write
        print(write(cfg.streaming.results_dir))


def _kafka_files(events: str, out: str, chunk: int, partitions: int) -> None:
    """Rows shaped like the Kafka source (key, value JSON, timestamp, partition, offset), in replay order."""
    import time

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(events)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    user, item = t["user"].to_numpy(), t["item"].to_numpy()
    ev, ts = t["event"].to_pylist(), t["timestamp"].to_numpy()
    part = (user % partitions).astype(np.int32)
    offsets = np.zeros(len(user), np.int64)
    seen = np.zeros(partitions, np.int64)
    for p in range(partitions):                       # per-partition offsets, like Kafka
        m = part == p
        offsets[m] = np.arange(m.sum())
        seen[p] = m.sum()
    now_ms = int(time.time() * 1000)
    for i, s in enumerate(range(0, len(user), chunk)):
        e = min(s + chunk, len(user))
        val = [json.dumps({"user": int(user[k]), "item": int(item[k]), "event": ev[k], "ts": int(ts[k])},
                          separators=(",", ":")).encode() for k in range(s, e)]
        key = [str(int(u)).encode() for u in user[s:e]]
        tbl = pa.table({"key": pa.array(key, pa.binary()), "value": pa.array(val, pa.binary()),
                        "timestamp": pa.array(np.full(e - s, now_ms, "datetime64[ms]"), pa.timestamp("ms")),
                        "partition": pa.array(part[s:e]), "offset": pa.array(offsets[s:e])})
        pq.write_table(tbl, o / f"part-{i:05d}.parquet")
    print(f"wrote {len(user):,} Kafka-shaped rows in {i + 1} files -> {o}")


if __name__ == "__main__":
    main()
