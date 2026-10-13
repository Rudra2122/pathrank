"""SparkSession for the streaming path (local mode on the laptop) and lake paths.

Packages are pulled by spark.jars.packages on first run (~300 MB into ~/.ivy2):
  spark-sql-kafka-0-10  matches the installed PySpark version
  hadoop-aws            matches the Hadoop bundled with PySpark (3.4.1 for Spark 4.0), AWS SDK v2
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from ..utils import log


def _bundled_hadoop_version() -> str:
    import pyspark
    jars = Path(pyspark.__file__).parent / "jars"
    for j in jars.glob("hadoop-client-api-*.jar"):
        m = re.search(r"hadoop-client-api-(.+)\.jar", j.name)
        if m:
            return m.group(1)
    return "3.4.1"


def lake(cfg, *parts: str) -> str:
    """s3a://<bucket>/... (MinIO) or file:///abs/<local_lake>/... (sink=local)."""
    s = cfg.streaming
    if s.sink == "s3":
        return "/".join([f"s3a://{s.bucket}", *parts])
    root = Path(s.local_lake).resolve()
    return (root.joinpath(*parts)).as_uri()


def ensure_bucket(cfg) -> None:
    """Create the MinIO bucket if it doesn't exist (replaces an `mc mb` init container)."""
    import time
    from urllib.parse import urlparse

    from pyarrow import fs as pafs

    s = cfg.streaming
    u = urlparse(s.s3_endpoint)
    s3 = pafs.S3FileSystem(access_key=s.s3_access_key, secret_key=s.s3_secret_key,
                           endpoint_override=u.netloc, scheme=u.scheme or "http", region="us-east-1",
                           allow_bucket_creation=True)
    for attempt in range(30):                     # MinIO may still be starting
        try:
            if s3.get_file_info(s.bucket).type == pafs.FileType.NotFound:
                s3.create_dir(s.bucket)
                log.info("created bucket %s at %s", s.bucket, s.s3_endpoint)
            return
        except OSError as e:
            if attempt == 29:
                raise SystemExit(f"can't reach MinIO at {s.s3_endpoint} ({e}); is `make stream-up` running?")
            time.sleep(1)


def session(cfg, app: str, kafka: bool = False, shuffle: int | None = None):
    import pyspark
    from pyspark.sql import SparkSession

    s = cfg.streaming
    scala = "2.13" if pyspark.__version__.startswith("4") else "2.12"
    pk = []
    if kafka:
        pk.append(f"org.apache.spark:spark-sql-kafka-0-10_{scala}:{pyspark.__version__}")
    if s.sink == "s3":
        pk.append(f"org.apache.hadoop:hadoop-aws:{_bundled_hadoop_version()}")
    b = (SparkSession.builder.appName(app)
         .master(os.environ.get("SPARK_MASTER", "local[*]"))
         .config("spark.driver.memory", s.driver_memory)
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.shuffle.partitions", str(shuffle or s.shuffle_partitions))
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.ui.showConsoleProgress", "false"))
    if pk:
        b = b.config("spark.jars.packages", ",".join(pk))
    if s.sink == "s3":
        ensure_bucket(cfg)
        h = "spark.hadoop.fs.s3a."
        b = (b.config(h + "endpoint", s.s3_endpoint)
             .config(h + "endpoint.region", "us-east-1")
             .config(h + "access.key", s.s3_access_key)
             .config(h + "secret.key", s.s3_secret_key)
             .config(h + "path.style.access", "true")
             .config(h + "connection.ssl.enabled", str(s.s3_endpoint.startswith("https")).lower())
             .config(h + "impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
             .config(h + "aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"))
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info("spark %s, master=%s, packages=%s, sink=%s", spark.version, spark.sparkContext.master,
             pk or "-", s.sink)
    return spark
