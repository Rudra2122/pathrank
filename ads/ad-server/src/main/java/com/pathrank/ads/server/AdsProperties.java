package com.pathrank.ads.server;

import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/** Everything tunable lives under {@code ads.*} in application.yml. */
@ConfigurationProperties("ads")
public record AdsProperties(
        @DefaultValue("file:ads/data/campaigns.json") String campaignsPath,
        @DefaultValue("ad-events") String eventsTopic,
        @DefaultValue("50000") long reserveMicros,          // $0.05 minimum cost per click
        @DefaultValue("10000") long incrementMicros,        // $0.01 over the second price
        @DefaultValue({"2", "5"}) List<Integer> positions,  // sponsored positions in the final list
        @DefaultValue("3") int frequencyCap,                // max impressions of one ad per user...
        @DefaultValue("3600000") long frequencyWindowMs,    // ...per hour
        @DefaultValue("1.0") double valuePerApplyUsd,       // what one organic apply is worth, for slot blending
        @DefaultValue("1.0") double blendAlpha,
        @DefaultValue("thompson") String exploration,       // "thompson" or "none" (rank by mean CTR)
        @DefaultValue("500") long ctrRefreshMs,             // how often CTRs are pulled from Redis
        @DefaultValue("600000") long impressionTtlMs) {     // clicks after this are not charged

    public int[] positionsArray() {
        return positions.stream().mapToInt(Integer::intValue).sorted().toArray();
    }
}
