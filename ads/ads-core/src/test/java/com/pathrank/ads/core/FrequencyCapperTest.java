package com.pathrank.ads.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class FrequencyCapperTest {
    @Test
    void capsWithinWindowAndForgetsAfterIt() {
        var now = new AtomicLong(0);
        var f = new FrequencyCapper(2, 1_000, now::get);
        assertTrue(f.allowed(1, "ad"));
        f.record(1, "ad");
        f.record(1, "ad");
        assertFalse(f.allowed(1, "ad"));
        assertTrue(f.allowed(2, "ad"));   // other user unaffected
        assertTrue(f.allowed(1, "other")); // other ad unaffected
        now.set(1_001);
        assertTrue(f.allowed(1, "ad"));
    }

    @Test
    void evictionDropsOnlyFullyExpiredPairs() {
        var now = new AtomicLong(0);
        var f = new FrequencyCapper(3, 1_000, now::get);
        f.record(1, "a");
        now.set(900);
        f.record(2, "b");
        now.set(1_500);
        assertEquals(1, f.evictExpired());
        assertEquals(1, f.trackedPairs());
    }
}
