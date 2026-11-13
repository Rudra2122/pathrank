package com.pathrank.ads.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Expected values come from running pathrank/notify/policy.py sponsored_slot on the same inputs. */
class SlotBlenderTest {
    private static final double[] ORGANIC = {0.30, 0.20, 0.10, 0.05, 0.02};

    @Test
    void matchesPythonDefaultSlot() {
        var d = SlotBlender.decide(ORGANIC, 0.04, 2.0, 2);
        assertEquals(0.04, d.expectedRevenue(), 1e-12);
        assertEquals(-0.003139412120706009, d.relevanceLoss(), 1e-12);
        assertTrue(d.insert());
    }

    @Test
    void matchesPythonWithValueAndAlpha() {
        var d = SlotBlender.decide(new double[] {0.5, 0.4, 0.3}, 0.2, 1.0, 1, 2.0, 0.5);
        assertEquals(0.1261859507142915, d.expectedRevenue(), 1e-12);
        assertEquals(0.07618595071429157, d.relevanceLoss(), 1e-12);
        assertTrue(d.insert());
    }

    @Test
    void rejectsLowValueAdThatPushesDownStrongResults() {
        // a near-zero ad at the top of a strong list costs more relevance than it earns
        var d = SlotBlender.decide(new double[] {0.9, 0.8, 0.7}, 0.001, 0.05, 0);
        assertFalse(d.insert());
    }
}
