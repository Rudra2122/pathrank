/** Small, dependency-free statistics (same definitions as the Python SDK). */

export function mean(xs: readonly number[]): number | null {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

export function stderr(xs: readonly number[]): number | null {
  const n = xs.length;
  if (n < 2) return null;
  const m = mean(xs)!;
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (n - 1)) / Math.sqrt(n);
}

/** Seeded PRNG (mulberry32): bootstrap CIs are reproducible run to run. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Percentile bootstrap CI of the mean. */
export function bootstrapCi(xs: readonly number[], nBoot = 2000, alpha = 0.05, seed = 0): [number, number] | null {
  const n = xs.length;
  if (n === 0) return null;
  if (n === 1) return [xs[0]!, xs[0]!];
  const r = rng(seed);
  const means = new Float64Array(nBoot);
  for (let b = 0; b < nBoot; b++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += xs[Math.floor(r() * n)]!;
    means[b] = s / n;
  }
  means.sort();
  return [means[Math.floor((alpha / 2) * nBoot)]!, means[Math.min(nBoot - 1, Math.floor((1 - alpha / 2) * nBoot))]!];
}

export function percentile(xs: readonly number[], q: number): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const k = (s.length - 1) * q;
  const f = Math.floor(k);
  const c = Math.ceil(k);
  return f === c ? s[f]! : s[f]! + (s[c]! - s[f]!) * (k - f);
}
