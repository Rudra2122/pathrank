#!/usr/bin/env bash
# End-to-end ads benchmark on one machine. Everything in results/ads/ and results/RESULTS_ads.md
# comes from this script.
#
#   1. Kafka + Redis in Docker (same compose file as the streaming path), fresh Redis, ad-events topic
#   2. campaigns on real OLX jobs (synthetic if events.parquet isn't there)
#   3. build the Java modules, start the Flink CTR job, then the Spring Boot ad server
#   4. click simulator: real OLX users, hidden per-ad CTR, closes the Kafka -> Flink -> Redis loop
#   5. k6 load test on POST /ads/auction
#   6. report (+ Flink freshness from Redis)
#
# Knobs: EVENTS, N_ADS (100), REQUESTS (200000), CONCURRENCY (16), VUS (32), DURATION (60s), WINDOW_MS (5000), PY
#        CACHE (redis | valkey): which cache the Flink job writes CTRs to and the ad server reads them from.
#        CACHE=valkey writes to results/ads_valkey/ and results/RESULTS_ads_valkey.md instead.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-python}
EVENTS=${EVENTS:-data/raw/olx/events.parquet}
REQUESTS=${REQUESTS:-200000}
CONCURRENCY=${CONCURRENCY:-16}
VUS=${VUS:-32}
DURATION=${DURATION:-60s}
WINDOW_MS=${WINDOW_MS:-5000}
N_ADS=${N_ADS:-100}
BUDGET_SCALE=${BUDGET_SCALE:-1}
CACHE=${CACHE:-redis}
case "$CACHE" in
  redis)  CACHE_PORT=6379; RES=results/ads;        MD=results/RESULTS_ads.md ;;
  valkey) CACHE_PORT=6380; RES=results/ads_valkey; MD=results/RESULTS_ads_valkey.md ;;
  *) echo "CACHE must be redis or valkey"; exit 1 ;;
esac
COMPOSE="docker compose -f deploy/docker-compose.yml --profile streaming --profile ads"
FLINK_OPENS="--add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.time=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/java.text=ALL-UNNAMED --add-opens=java.sql/java.sql=ALL-UNNAMED"

mkdir -p "$RES" logs
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

echo "== 1. Kafka + $CACHE"
$COMPOSE up -d --wait kafka "$CACHE"
if [ "$CACHE" = valkey ]; then $COMPOSE exec -T valkey valkey-cli FLUSHALL >/dev/null
else $COMPOSE exec -T redis redis-cli FLUSHALL >/dev/null; fi
$COMPOSE exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 \
  --create --if-not-exists --topic ad-events --partitions 12 --replication-factor 1 >/dev/null

echo "== 2. campaigns"
if [ -f "$EVENTS" ]; then
  $PY ads/sim/make_campaigns.py --events "$EVENTS" --n-ads "$N_ADS" --budget-scale "$BUDGET_SCALE"
else
  echo "   $EVENTS not found, using synthetic events (numbers will say so)"
  $PY ads/sim/make_campaigns.py --synthetic --n-ads "$N_ADS" --budget-scale "$BUDGET_SCALE"
fi

echo "== 3. build + start Flink job and ad server"
(cd ads && mvn -B -q -DskipTests package)
# --from latest: ignore events left on the topic by earlier runs (Redis was just flushed)
java $FLINK_OPENS -jar ads/ctr-aggregator/target/ctr-aggregator.jar --window-ms "$WINDOW_MS" --from latest \
  --redis-port "$CACHE_PORT" \
  > logs/ctr-aggregator.log 2>&1 &
PIDS+=($!)
sleep 15   # let the Flink job start and get partition assignments before any event is produced
# security off here so these numbers stay comparable with the first runs; compare.sh measures auth cost
java -jar ads/ad-server/target/ad-server.jar --ads.security.enabled=false --spring.data.redis.port="$CACHE_PORT" \
  > logs/ad-server.log 2>&1 &
PIDS+=($!)
for i in $(seq 1 60); do
  curl -sf localhost:8090/actuator/health >/dev/null && break
  sleep 1
  [ "$i" = 60 ] && { echo "ad server did not come up, see logs/ad-server.log"; exit 1; }
done

echo "== 4. click simulator ($REQUESTS requests)"
$PY ads/sim/click_sim.py --requests "$REQUESTS" --concurrency "$CONCURRENCY" --settle $(( WINDOW_MS / 1000 * 2 + 5 )) \
  --out "$RES/sim.json"

echo "== 5. k6 load test ($VUS VUs, $DURATION)"
if command -v k6 >/dev/null; then
  # native k6 talks to localhost directly; Docker Desktop's network proxy on a Mac caps throughput
  # and ran out of connections at the end of the first run
  k6 run -e RUNNER=native -e BASE=http://localhost:8090 -e DATA="$PWD/ads/data/sim_inputs.json" \
    -e OUT="$PWD/$RES/k6_auction.json" -e VUS="$VUS" -e DURATION="$DURATION" ads/bench/auction.js
else
  echo "   (k6 not installed, running it in Docker; brew install k6 gives cleaner numbers)"
  docker run --rm -i --add-host=host.docker.internal:host-gateway \
    -v "$PWD/ads:/ads" -v "$PWD/$RES:/results" grafana/k6 run \
    -e BASE=http://host.docker.internal:8090 -e VUS="$VUS" -e DURATION="$DURATION" /ads/bench/auction.js
fi

echo "== 6. report"
$PY ads/sim/report.py --window-ms "$WINDOW_MS" --redis-port "$CACHE_PORT" --results "$RES" --out "$MD"
