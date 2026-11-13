package com.pathrank.ads.core;

import java.util.SplittableRandom;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

/**
 * Exploration for the auction: instead of ranking by the mean CTR, draw each ad's CTR from its
 * Beta(clicks + alpha, impressions - clicks + beta) posterior on every request (Thompson sampling).
 *
 * <p>Without this, the auction only ever shows the ads that already look best (usually the
 * highest bidders on the 2% prior), so most ads never get enough impressions for their CTR to be
 * learned. With it, an ad with few impressions has a wide posterior and sometimes samples high, so
 * it gets tried; an ad with lots of data samples close to its true rate. Exploration fades on its
 * own as the posteriors narrow.
 */
public final class ThompsonCtrSource implements CtrSource {
    private final Function<String, long[]> counts; // adId -> {impressions, clicks}
    private final double alpha;
    private final double beta;
    private final SplittableRandom seeded; // null = ThreadLocalRandom (server); set for tests

    public ThompsonCtrSource(Function<String, long[]> counts, double alpha, double beta) {
        this(counts, alpha, beta, null);
    }

    public ThompsonCtrSource(Function<String, long[]> counts, double alpha, double beta, Long seed) {
        this.counts = counts;
        this.alpha = alpha;
        this.beta = beta;
        this.seeded = seed == null ? null : new SplittableRandom(seed);
    }

    @Override
    public double pctr(String adId) {
        long[] c = counts.apply(adId);
        long imp = c == null ? 0 : c[0];
        long clk = c == null ? 0 : Math.min(c[1], imp);
        double x = gamma(clk + alpha);
        double y = gamma(imp - clk + beta);
        return x / (x + y);
    }

    private double uniform() {
        if (seeded != null) {
            synchronized (seeded) {
                return seeded.nextDouble();
            }
        }
        return ThreadLocalRandom.current().nextDouble();
    }

    private double normal() {
        // Box-Muller
        double u1 = Math.max(uniform(), 1e-300), u2 = uniform();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /** Marsaglia-Tsang gamma sampler (shape k > 0, scale 1). */
    double gamma(double k) {
        if (k < 1) return gamma(k + 1) * Math.pow(uniform(), 1.0 / k);
        double d = k - 1.0 / 3, c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x = normal(), v = 1 + c * x;
            if (v <= 0) continue;
            v = v * v * v;
            double u = uniform();
            if (u < 1 - 0.0331 * x * x * x * x) return d * v;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
    }
}
