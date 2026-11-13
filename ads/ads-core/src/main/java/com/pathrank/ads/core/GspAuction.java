package com.pathrank.ads.core;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Generalized second-price (GSP) auction with quality: ads are ranked by bid x pCTR (expected
 * cost per impression), and the winner in position i pays the smallest per-click price that would
 * still have kept it above position i+1:
 *
 * <pre>  price_i = max(reserve, rankScore_{i+1} / pCTR_i) + increment,  capped at bid_i</pre>
 *
 * Ranking by bid x pCTR instead of bid alone means a high bid can't buy the slot for an ad nobody
 * clicks. Ties break on adId so the result is deterministic.
 */
public final class GspAuction {
    public record Candidate(Campaign campaign, double pctr) {
        public double rankScore() {
            return campaign.bidMicros() * pctr;
        }
    }

    public record Winner(Campaign campaign, double pctr, long priceMicros, double rankScore) {}

    private final long reserveMicros;
    private final long incrementMicros;

    public GspAuction(long reserveMicros, long incrementMicros) {
        if (reserveMicros < 0 || incrementMicros < 0) throw new IllegalArgumentException("reserve/increment must be >= 0");
        this.reserveMicros = reserveMicros;
        this.incrementMicros = incrementMicros;
    }

    public List<Winner> run(List<Candidate> candidates, int positions) {
        List<Candidate> eligible = new ArrayList<>();
        for (Candidate c : candidates) {
            if (c.pctr() > 0 && c.campaign().bidMicros() >= reserveMicros) eligible.add(c);
        }
        eligible.sort(Comparator.comparingDouble(Candidate::rankScore).reversed()
                .thenComparing(c -> c.campaign().adId()));

        int k = Math.min(positions, eligible.size());
        List<Winner> out = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            Candidate c = eligible.get(i);
            long price;
            if (i + 1 < eligible.size()) {
                double next = eligible.get(i + 1).rankScore();
                price = (long) Math.ceil(next / c.pctr()) + incrementMicros;
                price = Math.max(price, reserveMicros);
            } else {
                price = reserveMicros;
            }
            price = Math.min(price, c.campaign().bidMicros());
            out.add(new Winner(c.campaign(), c.pctr(), price, c.rankScore()));
        }
        return out;
    }

    public long reserveMicros() {
        return reserveMicros;
    }
}
