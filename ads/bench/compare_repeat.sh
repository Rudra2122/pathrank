#!/usr/bin/env bash
# Same comparison as compare.sh, but each config 3x in interleaved order with cool-downs; reports medians.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${ROUNDS:-3}; DURATION=${DURATION:-30s}; COOL=${COOL:-45}
OUT=results/ads/compare_repeat; mkdir -p "$OUT" logs
SERVER=""
stop() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null && wait "$SERVER" 2>/dev/null || true; SERVER=""; }
trap stop EXIT
start() {
  java -jar ads/ad-server/target/ad-server.jar "$@" > logs/ad-server-repeat.log 2>&1 & SERVER=$!
  for i in $(seq 1 60); do curl -sf localhost:8090/actuator/health >/dev/null && return 0; sleep 1; done
  echo "server did not come up"; exit 1
}
run() {  # run <label> <vus> <auth> <server args...>
  local label=$1 vus=$2 auth=$3; shift 3
  start "$@"
  local c=(-e RUNNER=native -e BASE=http://localhost:8090 -e DATA="$PWD/ads/data/sim_inputs.json" -e AUTH="$auth")
  k6 run -q "${c[@]}" -e OUT=/dev/null -e VUS="$vus" -e DURATION=10s ads/bench/auction.js >/dev/null
  k6 run -q "${c[@]}" -e OUT="$PWD/$OUT/${label}_r${r}.json" -e LABEL="$label" -e VUS="$vus" -e DURATION="$DURATION" ads/bench/auction.js >/dev/null
  stop; echo "  $label r$r done"; sleep "$COOL"
}
CONFIGS=(
  "platform_vus32_noauth 32 off --ads.security.enabled=false --spring.threads.virtual.enabled=false"
  "virtual_vus32_noauth 32 off --ads.security.enabled=false --spring.threads.virtual.enabled=true"
  "platform_vus32_auth 32 on --ads.security.enabled=true --spring.threads.virtual.enabled=false"
  "platform_vus256_noauth 256 off --ads.security.enabled=false --spring.threads.virtual.enabled=false"
  "virtual_vus256_noauth 256 off --ads.security.enabled=false --spring.threads.virtual.enabled=true"
  "platform_vus256_auth 256 on --ads.security.enabled=true --spring.threads.virtual.enabled=false"
)
for r in $(seq 1 "$ROUNDS"); do
  echo "== round $r"
  # rotate the starting point each round so no config always runs on a hot CPU
  n=${#CONFIGS[@]}
  for i in $(seq 0 $((n-1))); do
    read -r -a cfg <<< "${CONFIGS[$(( (i + r) % n ))]}"
    run "${cfg[@]}"
  done
done
python3 - <<'PY'
import glob, json, os, statistics as st, collections
runs = collections.defaultdict(list)
for p in glob.glob("results/ads/compare_repeat/*_r*.json"):
    d = json.load(open(p)); runs[d["label"]].append(d)
out = {}
print(f"{'config':26} {'rounds':>6} {'req/s median (min-max)':>26} {'p99 ms median':>14}")
for k in sorted(runs):
    v = runs[k]; rps = [x["req_per_s"] for x in v]; p99 = [x["latency_ms"]["p99"] for x in v]
    out[k] = {"rounds": len(v), "req_per_s_median": st.median(rps), "req_per_s_min": min(rps),
              "req_per_s_max": max(rps), "p99_ms_median": st.median(p99), "vus": v[0]["vus"], "auth": v[0]["auth"]}
    print(f"{k:26} {len(v):>6} {st.median(rps):>10,.0f} ({min(rps):,.0f}-{max(rps):,.0f}) {st.median(p99):>14.1f}")
json.dump(out, open("results/ads/compare_repeat/summary.json", "w"), indent=2)
PY
