package com.pathrank.ads.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class ThompsonCtrSourceTest {
    private static double[] meanAndSd(ThompsonCtrSource s, String ad, int n) {
        double sum = 0, sq = 0;
        for (int i = 0; i < n; i++) { double x = s.pctr(ad); sum += x; sq += x * x; }
        double m = sum / n;
        return new double[] {m, Math.sqrt(sq / n - m * m)};
    }

    @Test
    void samplesAverageToThePosteriorMean() {
        var s = new ThompsonCtrSource(id -> new long[] {10_000, 500}, 1, 49, 42L);
        double[] ms = meanAndSd(s, "a", 20_000);
        assertEquals(CtrEstimate.smoothed(500, 10_000), ms[0], 0.0005);
    }

    @Test
    void newAdsExploreMoreThanWellKnownAds() {
        Map<String, long[]> c = Map.of("known", new long[] {50_000, 1_000});
        var s = new ThompsonCtrSource(c::get, 1, 49, 7L);
        double sdNew = meanAndSd(s, "new", 20_000)[1];
        double sdKnown = meanAndSd(s, "known", 20_000)[1];
        assertTrue(sdNew > 10 * sdKnown, "new " + sdNew + " known " + sdKnown);
        assertEquals(0.02, meanAndSd(s, "new", 20_000)[0], 0.002); // prior mean
    }

    @Test
    void gammaSamplerHasTheRightMean() {
        var s = new ThompsonCtrSource(id -> null, 1, 1, 3L);
        for (double k : new double[] {0.5, 1, 3, 50}) {
            double sum = 0;
            for (int i = 0; i < 50_000; i++) sum += s.gamma(k);
            assertEquals(k, sum / 50_000, k * 0.03);
        }
    }
}
