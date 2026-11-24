package com.pathrank.ads.server;

import com.pathrank.ads.core.BudgetStore;
import com.pathrank.ads.core.FrequencyCapper;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/** Background cleanup: expire old impressions and frequency-cap state, reset budgets each UTC day. */
@Component
public class Maintenance {
    private final ImpressionCache impressions;
    private final FrequencyCapper frequency;
    private final BudgetStore budget;
    private final AdsProperties props;

    public Maintenance(ImpressionCache impressions, FrequencyCapper frequency, BudgetStore budget, AdsProperties props) {
        this.impressions = impressions;
        this.frequency = frequency;
        this.budget = budget;
        this.props = props;
    }

    @Scheduled(fixedDelay = 30_000)
    public void evict() {
        impressions.evictOlderThan(System.currentTimeMillis() - props.impressionTtlMs());
        frequency.evictExpired();
    }

    @Scheduled(cron = "0 0 0 * * *", zone = "UTC")
    public void newBudgetDay() {
        budget.resetAll();
    }
}
