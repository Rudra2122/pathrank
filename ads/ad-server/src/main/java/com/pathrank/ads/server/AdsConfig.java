package com.pathrank.ads.server;

import com.pathrank.ads.core.AdDecisionEngine;
import com.pathrank.ads.core.BudgetLedger;
import com.pathrank.ads.core.CtrEstimate;
import com.pathrank.ads.core.CtrSource;
import com.pathrank.ads.core.FrequencyCapper;
import com.pathrank.ads.core.GspAuction;
import com.pathrank.ads.core.ThompsonCtrSource;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** Wires the framework-free ads-core classes into Spring. */
@Configuration
public class AdsConfig {
    @Bean
    BudgetLedger budgetLedger() {
        return new BudgetLedger();
    }

    @Bean
    FrequencyCapper frequencyCapper(AdsProperties p) {
        return new FrequencyCapper(p.frequencyCap(), p.frequencyWindowMs(), System::currentTimeMillis);
    }

    @Bean
    ImpressionCache impressionCache() {
        return new ImpressionCache();
    }

    @Bean
    AdStats adStats() {
        return new AdStats();
    }

    /** CTR used to rank the auction: a Thompson sample from the posterior (default), or the mean. */
    static CtrSource auctionCtr(AdsProperties p, RedisCtrSource ctr) {
        return switch (p.exploration()) {
            case "thompson" -> new ThompsonCtrSource(ctr::rawCounts, CtrEstimate.DEFAULT_ALPHA, CtrEstimate.DEFAULT_BETA);
            case "none" -> ctr;
            default -> throw new IllegalArgumentException("ads.exploration must be thompson or none, got " + p.exploration());
        };
    }

    @Bean
    AdDecisionEngine adDecisionEngine(AdsProperties p, BudgetLedger budget, FrequencyCapper freq, RedisCtrSource ctr) {
        return new AdDecisionEngine(new GspAuction(p.reserveMicros(), p.incrementMicros()), budget, freq,
                auctionCtr(p, ctr), p.valuePerApplyUsd(), p.blendAlpha());
    }

    @Bean
    AdService adService(AdsProperties p, AdDecisionEngine engine, CampaignStore campaigns, BudgetLedger budget,
                        ImpressionCache impressions, AdEventPublisher events, AdStats stats, MeterRegistry registry) {
        return new AdService(engine, campaigns, budget, impressions, events, stats, p.positionsArray(),
                p.impressionTtlMs(), System::currentTimeMillis, registry);
    }
}
