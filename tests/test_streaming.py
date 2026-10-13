"""Streaming path: lag percentiles (no Spark), and with PySpark + Java: Spark vs pandas feature parity,
streaming job (Kafka-shaped file source) vs batch windows. Skipped when pyspark or java is missing."""
import shutil
import subprocess
import sys
from collections import Counter

import pytest

from pathrank.streaming.stream import percentiles

ROOT_CFG = "configs/streaming_smoke.yaml"


def test_percentiles_nearest_rank():
    h = Counter({10: 50, 20: 45, 100: 4, 500: 1})       # 100 rows
    p = percentiles(h)
    assert p["p50"] == 10 and p["p95"] == 20 and p["p99"] == 100 and p["max"] == 500
    assert percentiles(Counter()) == {}


spark_ok = shutil.which("java") is not None
try:
    import pyspark  # noqa: F401
except ImportError:
    spark_ok = False
needs_spark = pytest.mark.skipif(not spark_ok, reason="needs pyspark + java 17+")


def _run(*args):
    r = subprocess.run([sys.executable, "scripts/streaming.py", *args, "--config", ROOT_CFG],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return r


@pytest.fixture(scope="module")
def events(tmp_path_factory):
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    shutil.rmtree("data/streaming_smoke", ignore_errors=True)
    _run("synth", "--n", "60000")
    return "data/streaming_smoke/events.parquet"


@needs_spark
def test_spark_batch_matches_pandas_asof(events):
    import json
    _run("batch-parity")
    r = json.load(open("data/streaming_smoke/results/batch_parity.json"))
    assert r["pass"] and r["max_abs_diff"] == 0 and r["pairs"] == r["pairs_spark"]


@needs_spark
def test_stream_windows_match_batch(events):
    import json
    _run("kafka-files", "--out", "data/streaming_smoke/kafka_files", "--chunk", "20000")
    _run("stream", "--source", "files", "--files-dir", "data/streaming_smoke/kafka_files", "--reset",
         "--idle", "12")
    s = json.load(open("data/streaming_smoke/results/stream.json"))
    assert s["events"] == 60000 and s["batches"] == 3
    _run("stream-parity")
    r = json.load(open("data/streaming_smoke/results/stream_parity.json"))
    assert r["pass"], r
    assert r["bronze"] == {"rows": 60000, "duplicates": 0}
    assert r["job_popularity"]["matched"] > 0 and r["job_popularity"]["missing_closed"] == 0
