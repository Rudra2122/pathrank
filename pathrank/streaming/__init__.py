"""Streaming path: OLX events -> Kafka (Java producer) -> Spark Structured Streaming -> S3 (MinIO).

events.py   raw OLX events as one timestamp-sorted Parquet file (what the producer replays)
spark.py    SparkSession with the Kafka + S3A (MinIO) packages, lake paths
stream.py   the streaming job: bronze events + windowed job popularity / user activity, lag capture
batch.py    Spark batch: the same windowed features over the full Parquet, and the as-of features
parity.py   checks: Spark batch vs the pandas AsOf features, stream vs batch windows, Kafka -> S3 counts
report.py   results/streaming/*.json -> results/RESULTS_streaming.md + the README table
"""
CONTACT_EVENTS = ("chat_click", "contact_phone_click_1", "contact_phone_click_2", "contact_phone_click_3",
                  "contact_partner_click", "contact_chat")
