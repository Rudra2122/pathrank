// k6 load test for POST /ads/auction. Run through Docker so nothing needs installing:
//   docker run --rm -i -v "$PWD/ads:/ads" -v "$PWD/results/ads:/results" grafana/k6 run \
//     -e BASE=http://host.docker.internal:8090 -e VUS=32 -e DURATION=60s /ads/bench/auction.js
// Faster and more accurate without Docker's network proxy: brew install k6 (the bench uses it if present).
// Each request: a real OLX user id and 10 organic jobs from the popular pool (same as the click sim).
import http from 'k6/http';
import { check } from 'k6';

const inputs = JSON.parse(open(__ENV.DATA || '/ads/data/sim_inputs.json'));
const BASE = __ENV.BASE || 'http://host.docker.internal:8090';
const OUT = __ENV.OUT || '/results/k6_auction.json';
// AUTH=on: log in once as the SERVICE user and send the JWT on every request (security on).
const AUTH = (__ENV.AUTH || 'off') === 'on';
const LABEL = __ENV.LABEL || '';

export const options = {
  vus: parseInt(__ENV.VUS || '32'),
  duration: __ENV.DURATION || '60s',
  summaryTrendStats: ['avg', 'p(50)', 'p(95)', 'p(99)', 'max'],
};

function sample(arr, k) {
  const out = [];
  const used = new Set();
  while (out.length < k) {
    const i = Math.floor(Math.random() * arr.length);
    if (!used.has(i)) { used.add(i); out.push(arr[i]); }
  }
  return out.sort((a, b) => b.score - a.score);
}

export function setup() {
  if (!AUTH) return { token: null };
  const r = http.post(`${BASE}/auth/token`, JSON.stringify({ username: __ENV.SERVICE_USER || 'jobboard',
    password: __ENV.SERVICE_PASSWORD || 'jobboard-pass' }), { headers: { 'Content-Type': 'application/json' } });
  if (r.status !== 200) throw new Error(`login failed: ${r.status}`);
  return { token: r.json('accessToken') };
}

export default function (ctx) {
  const body = JSON.stringify({
    userId: inputs.users[Math.floor(Math.random() * inputs.users.length)],
    organic: sample(inputs.pool, 10),
  });
  const headers = { 'Content-Type': 'application/json' };
  if (ctx.token) headers.Authorization = `Bearer ${ctx.token}`;
  const r = http.post(`${BASE}/ads/auction`, body, { headers });
  check(r, { 'status 200': (x) => x.status === 200 });
}

export function handleSummary(data) {
  const d = data.metrics.http_req_duration.values;
  const out = {
    label: LABEL,
    auth: AUTH,
    runner: __ENV.RUNNER || 'docker',
    vus: options.vus,
    duration: options.duration,
    requests: data.metrics.http_reqs.values.count,
    req_per_s: data.metrics.http_reqs.values.rate,
    failed_rate: data.metrics.http_req_failed.values.rate,
    latency_ms: { avg: d.avg, p50: d['p(50)'], p95: d['p(95)'], p99: d['p(99)'], max: d.max },
  };
  return { [OUT]: JSON.stringify(out, null, 2), stdout: JSON.stringify(out, null, 2) + '\n' };
}
