# Streaming lag sweep: trigger interval and replay rate

First 5,000,000 OLX events (`data/raw/olx/events_head5000000.parquet`, about 30 h of event time), Kafka (KRaft, 12 partitions)
→ Spark Structured Streaming 4.0.1 (local[*], 8 cores) → MinIO. Three queries: bronze (lag measured here),
job popularity 1 h, user activity 1 h. MacBook Air M3, Docker Desktop, Java 17.
Lag = time the bronze Parquet write to S3 returned − Kafka record timestamp (producer send time), exact 1 ms histogram.

| Run | Trigger | Replay rate (producer, measured) | Spark processed | Batches | p50 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| `results/streaming` | 5 s | 49,992/s | 48,412/s | 21 | 5,861 ms | 8,715 ms | 9,479 ms | 9,824 ms | 5,868 ms |
| `results/streaming_1s` | 1 s | ~50,000/s | 48,989/s | 89 | 1,464 ms | 4,046 ms | 5,301 ms | 6,288 ms | 1,763 ms |
| `results/streaming_150k` | 1 s | ~150,000/s | 127,918/s | 8 | 7,192 ms | 10,509 ms | 11,330 ms | 11,663 ms | 7,307 ms |

"Spark processed" = rows / (first batch start → last batch commit). At 150K/s Spark fell behind, so lag grows with run
length. That run lasted about 33 s, so its lag numbers are a lower bound and ~128K/s is an approximate ceiling.

Reproduce one row (for example 1 s, 150K/s):

```bash
JAR=streaming/producer/target/olx-replay-producer.jar; OUT=results/streaming_150k; mkdir -p $OUT
java -jar $JAR --topic job-events --partitions 12 --recreate-topic --create-only
python scripts/streaming.py stream --config configs/olx_streaming.yaml --set streaming.trigger="1 second" streaming.results_dir=$OUT --reset --producer-rate 150000 --idle 30 &
until [ -f $OUT/stream.ready ]; do sleep 2; done
java -jar $JAR --parquet data/raw/olx/events_head5000000.parquet --topic job-events --rate 150000 --out $OUT/producer_lag.json
wait
```
