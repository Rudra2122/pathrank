package com.pathrank.ads.core;

/**
 * Java port of {@code pathrank.notify.policy.sponsored_slot}: insert a sponsored job at a slot only
 * if its expected revenue beats the change in exposure-weighted relevance of the organic list.
 *
 * <p>Exposure weight of position i is 1 / log2(i + 2) (the NDCG discount). Scores are treated as
 * apply probabilities. {@code relevanceLoss} = alpha * valuePerApply * (relevance before - after),
 * where "after" is the list with the sponsored job inserted and truncated back to the same length.
 * The Python and Java versions are checked against the same numbers in SlotBlenderTest.
 */
public final class SlotBlender {
    private SlotBlender() {}

    public record Decision(boolean insert, double expectedRevenue, double relevanceLoss) {}

    public static Decision decide(double[] organic, double sponsoredScore, double bid, int slot,
                                  double valuePerApply, double alpha) {
        int n = organic.length;
        if (slot < 0 || slot > n) throw new IllegalArgumentException("slot out of range: " + slot);
        double[] w = new double[n + 1];
        for (int i = 0; i < w.length; i++) w[i] = 1.0 / (Math.log(i + 2) / Math.log(2));

        double before = 0;
        for (int i = 0; i < n; i++) before += organic[i] * w[i];

        double[] after = new double[n];
        for (int i = 0, src = 0; i < n; i++) {
            after[i] = (i == slot) ? sponsoredScore : organic[src++];
        }
        double afterSum = 0;
        for (int i = 0; i < n; i++) afterSum += after[i] * w[i];

        double loss = alpha * valuePerApply * (before - afterSum);
        double revenue = bid * sponsoredScore * w[slot];
        return new Decision(revenue > loss, revenue, loss);
    }

    public static Decision decide(double[] organic, double sponsoredScore, double bid, int slot) {
        return decide(organic, sponsoredScore, bid, slot, 1.0, 1.0);
    }
}
