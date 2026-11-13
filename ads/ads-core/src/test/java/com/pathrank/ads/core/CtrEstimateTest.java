package com.pathrank.ads.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class CtrEstimateTest {
    @Test
    void newAdGetsThePriorMean() {
        assertEquals(0.02, CtrEstimate.smoothed(0, 0), 1e-12);
    }

    @Test
    void movesTowardObservedRateWithData() {
        double est = CtrEstimate.smoothed(500, 10_000); // observed 5%
        assertEquals((500 + 1.0) / (10_000 + 50.0), est, 1e-12);
        assertEquals(0.05, est, 0.001);
    }

    @Test
    void clicksAheadOfImpressionsAreClamped() {
        assertEquals((3 + 1.0) / (3 + 50.0), CtrEstimate.smoothed(5, 3), 1e-12);
    }
}
