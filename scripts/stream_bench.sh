#!/usr/bin/env bash
# End-to-end streaming benchmark: the four numbers in the README streaming table.
#
# Prereqs (see streaming/README.md):
#   make stream-up                 Kafka (KRaft) + MinIO in Docker
#   make stream-producer-build     Java producer fat jar (Maven, JDK 17+)
#   make stream-events             data/raw/olx/events.parquet from the Kaggle CSV
#
# Knobs: EVENTS (full replay file), LAG_N (events in the lag run), RATE (lag-run replay rate),
#        CFG, PY. Results land in results/streaming/*.json, then the report fills the README table.
set -euo pipefail
CFG=${CFG:-configs/olx_streaming.yaml}
EVENTS=${EVENTS:-data/raw/olx/events.parquet}
LAG_N=${LAG_N:-5000000}
LAG_EVENTS=${LAG_EVENTS:-${EVENTS%.parquet}_head${LAG_N}.parquet}
RATE=${RATE:-50000}
PARTITIONS=${PARTITIONS:-12}
PY=${PY:-python}
JAR=streaming/producer/target/olx-replay-producer.jar
OUT=results/streaming
mkdir -p "$OUT"
[ -f "$JAR" ] || { echo "missing $JAR: run make stream-producer-build"; exit 1; }
[ -f "$EVENTS" ] || { echo "missing $EVENTS: run make stream-events"; exit 1; }

echo "== 1/5 producer throughput: full replay at max rate (topic job-events-bench)"
java -jar "$JAR" --parquet "$EVENTS" --dry-run --out "$OUT/producer_dryrun.json" >/dev/null
java -jar "$JAR" --parquet "$EVENTS" --topic job-events-bench --partitions "$PARTITIONS" \
     --recreate-topic --rate 0 --out "$OUT/producer.json"

echo "== 2/5 lag run: $LAG_N events at $RATE events/s, Kafka -> Spark -> MinIO"
[ -f "$LAG_EVENTS" ] || $PY scripts/streaming.py head --config "$CFG" --events "$EVENTS" --n "$LAG_N" --out "$LAG_EVENTS"
java -jar "$JAR" --topic job-events --partitions "$PARTITIONS" --recreate-topic --create-only
rm -f "$OUT/stream.ready"
$PY scripts/streaming.py stream --config "$CFG" --reset --producer-rate "$RATE" --idle 30 &
SPID=$!
trap 'kill $SPID 2>/dev/null || true' EXIT
until [ -f "$OUT/stream.ready" ]; do
  sleep 2
  kill -0 $SPID 2>/dev/null || { echo "stream job exited before it was ready"; exit 1; }
done
java -jar "$JAR" --parquet "$LAG_EVENTS" --topic job-events --partitions "$PARTITIONS" \
     --rate "$RATE" --out "$OUT/producer_lag.json"
wait $SPID
trap - EXIT

status=0
echo "== 3/5 stream vs batch windows, Kafka -> S3 completeness"
$PY scripts/streaming.py stream-parity --config "$CFG" --events "$LAG_EVENTS" || { echo "stream parity FAILED"; status=1; }
echo "== 4/5 Spark batch vs pandas, full events"
$PY scripts/streaming.py batch-parity --config "$CFG" --events "$EVENTS" || { echo "batch parity FAILED"; status=1; }
echo "== 5/5 report"
$PY scripts/streaming.py report --config "$CFG"
exit $status
