package com.pathrank.ads.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class BudgetLedgerTest {
    @Test
    void refusesChargeThatWouldExceedBudget() {
        var l = new BudgetLedger();
        assertTrue(l.charge("a", 1_000_000, 600_000));
        assertFalse(l.charge("a", 1_000_000, 500_000));
        assertEquals(600_000, l.spent("a"));
        assertFalse(l.canAfford("a", 1_000_000, 500_000));
        assertTrue(l.canAfford("a", 1_000_000, 400_000));
    }

    @Test
    void concurrentClicksNeverOverspend() throws Exception {
        var l = new BudgetLedger();
        long budget = 1_000_000, price = 1_000; // room for exactly 1,000 clicks
        var ok = new AtomicInteger();
        ExecutorService pool = Executors.newFixedThreadPool(8);
        for (int i = 0; i < 5_000; i++) pool.submit(() -> { if (l.charge("a", budget, price)) ok.incrementAndGet(); });
        pool.shutdown();
        assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS));
        assertEquals(1_000, ok.get());
        assertEquals(budget, l.spent("a"));
    }

    @Test
    void resetStartsANewDay() {
        var l = new BudgetLedger();
        l.charge("a", 10, 10);
        l.resetAll();
        assertEquals(0, l.spent("a"));
    }
}
