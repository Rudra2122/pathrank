package com.pathrank.ads.core;

/**
 * Where daily advertiser spend lives. {@link BudgetLedger} keeps it in memory (one ad server);
 * the ad server's JPA store keeps it in PostgreSQL or CockroachDB so several ad servers can
 * charge the same advertiser without overspending.
 */
public interface BudgetStore {
    /** True if the advertiser could still pay {@code maxPriceMicros} today. Used to filter auction candidates. */
    boolean canAfford(String advertiserId, long dailyBudgetMicros, long maxPriceMicros);

    /** Charge a click. Returns false (and charges nothing) if it would exceed the budget. */
    boolean charge(String advertiserId, long dailyBudgetMicros, long priceMicros);

    long spent(String advertiserId);

    /** Called at the start of each budget day. */
    void resetAll();
}
