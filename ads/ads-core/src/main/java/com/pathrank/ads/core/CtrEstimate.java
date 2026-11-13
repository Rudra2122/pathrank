package com.pathrank.ads.core;

/**
 * Beta-smoothed click-through rate: (clicks + alpha) / (impressions + alpha + beta).
 *
 * <p>A new ad with 0 impressions gets the prior mean alpha / (alpha + beta) instead of 0 or NaN,
 * and the estimate moves toward the observed rate as impressions grow. With alpha=1, beta=49 the
 * prior is 2%, and after ~1,000 impressions the data dominates.
 */
public final class CtrEstimate {
    public static final double DEFAULT_ALPHA = 1.0;
    public static final double DEFAULT_BETA = 49.0;

    private CtrEstimate() {}

    public static double smoothed(long clicks, long impressions, double alpha, double beta) {
        if (clicks < 0 || impressions < 0) throw new IllegalArgumentException("counts must be >= 0");
        if (alpha <= 0 || beta <= 0) throw new IllegalArgumentException("prior must be > 0");
        // clicks can briefly exceed impressions when a click event lands in an earlier window than
        // its impression; clamp so the rate stays a probability
        long c = Math.min(clicks, impressions);
        return (c + alpha) / (impressions + alpha + beta);
    }

    public static double smoothed(long clicks, long impressions) {
        return smoothed(clicks, impressions, DEFAULT_ALPHA, DEFAULT_BETA);
    }
}
