package com.pathrank.ads.core;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Daily spend per advertiser. Clicks are charged with a compare-and-set loop, so two clicks
 * arriving at once can never push an advertiser over budget.
 */
public final class BudgetLedger implements BudgetStore {
    private final Map<String, AtomicLong> spent = new ConcurrentHashMap<>();

    /** True if the advertiser could still pay {@code maxPriceMicros} today. Used to filter auction candidates. */
    @Override
    public boolean canAfford(String advertiserId, long dailyBudgetMicros, long maxPriceMicros) {
        return spent(advertiserId) + maxPriceMicros <= dailyBudgetMicros;
    }

    /** Charge a click. Returns false (and charges nothing) if it would exceed the budget. */
    @Override
    public boolean charge(String advertiserId, long dailyBudgetMicros, long priceMicros) {
        if (priceMicros < 0) throw new IllegalArgumentException("price must be >= 0");
        AtomicLong s = spent.computeIfAbsent(advertiserId, k -> new AtomicLong());
        while (true) {
            long cur = s.get();
            long next = cur + priceMicros;
            if (next > dailyBudgetMicros) return false;
            if (s.compareAndSet(cur, next)) return true;
        }
    }

    @Override
    public long spent(String advertiserId) {
        AtomicLong s = spent.get(advertiserId);
        return s == null ? 0L : s.get();
    }

    public Map<String, Long> snapshot() {
        Map<String, Long> out = new java.util.TreeMap<>();
        spent.forEach((k, v) -> out.put(k, v.get()));
        return out;
    }

    /** Called at the start of each budget day. */
    @Override
    public void resetAll() {
        spent.clear();
    }
}
