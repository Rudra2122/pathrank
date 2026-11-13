package com.pathrank.ads.core;

/**
 * One sponsored-job campaign: an advertiser pays per click to promote one job.
 * Money is in micros (1 USD = 1,000,000) so nothing is ever rounded through a double.
 */
public record Campaign(String adId, String advertiserId, long jobId, long bidMicros, long dailyBudgetMicros) {
    public Campaign {
        if (adId == null || adId.isBlank()) throw new IllegalArgumentException("adId is required");
        if (advertiserId == null || advertiserId.isBlank()) throw new IllegalArgumentException("advertiserId is required");
        if (bidMicros <= 0) throw new IllegalArgumentException("bid must be > 0");
        if (dailyBudgetMicros <= 0) throw new IllegalArgumentException("daily budget must be > 0");
    }
}
