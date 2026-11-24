#!/usr/bin/env bash
# Serving-path comparisons on one machine, after `make ads-bench` has made ads/data/:
#   1. platform threads vs Java 21 virtual threads, at a normal and a high number of concurrent users
#   2. auth off vs on (Spring Security JWT check on every auction request)
# then a live authorization probe against the secured server (security_probe.py).
# Each run: start the ad server with one setting, 10 s k6 warm-up (thrown away), measured k6 run,
# stop the server. Output: results/ads/compare/*.json, results/ads/security.json.
#
# Knobs: VUS_LIST ("32 256"), DURATION (30s), PY
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-python}
VUS_LIST=${VUS_LIST:-"32 256"}
DURATION=${DURATION:-30s}
COMPOSE="docker compose -f deploy/docker-compose.yml --profile streaming --profile ads"
OUT=results/ads/compare
mkdir -p "$OUT" logs
[ -f ads/data/sim_inputs.json ] || { echo "run make ads-bench (or make ads-campaigns) first"; exit 1; }
command -v k6 >/dev/null || { echo "needs native k6 (brew install k6): Docker's network proxy caps throughput"; exit 1; }

SERVER=""
stop() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null && wait "$SERVER" 2>/dev/null || true; SERVER=""; }
trap stop EXIT

start() {  # start <extra args...>
  java -jar ads/ad-server/target/ad-server.jar "$@" > logs/ad-server-compare.log 2>&1 &
  SERVER=$!
  for i in $(seq 1 60); do
    curl -sf localhost:8090/actuator/health >/dev/null && return 0
    sleep 1
  done
  echo "ad server did not come up, see logs/ad-server-compare.log"; exit 1
}

k6run() {  # k6run <label> <vus> <auth on|off>
  local common=(-e RUNNER=native -e BASE=http://localhost:8090 -e DATA="$PWD/ads/data/sim_inputs.json" -e AUTH="$3")
  k6 run -q "${common[@]}" -e OUT=/dev/null -e VUS="$2" -e DURATION=10s ads/bench/auction.js >/dev/null
  k6 run -q "${common[@]}" -e OUT="$PWD/$OUT/$1.json" -e LABEL="$1" -e VUS="$2" -e DURATION="$DURATION" ads/bench/auction.js
}

echo "== Kafka + Redis"
$COMPOSE up -d --wait kafka redis
(cd ads && mvn -B -q -DskipTests package)

for threads in platform virtual; do
  vt=false; [ "$threads" = virtual ] && vt=true
  echo "== $threads threads, auth off"
  start --ads.security.enabled=false --spring.threads.virtual.enabled=$vt
  for v in $VUS_LIST; do k6run "${threads}_vus${v}_noauth" "$v" off; done
  stop
done

echo "== platform threads, auth on (JWT verified on every request)"
start --ads.security.enabled=true --spring.threads.virtual.enabled=false
for v in $VUS_LIST; do k6run "platform_vus${v}_auth" "$v" on; done
echo "== authorization probe against the live secured server"
$PY ads/bench/security_probe.py --out results/ads/security.json
stop

echo "== done: $OUT/*.json, results/ads/security.json. make ads-v2-report writes results/RESULTS_ads_v2.md"
