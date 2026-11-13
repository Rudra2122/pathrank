package com.pathrank.ads.flink;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Test;

class CountAggregateTest {
    private static AdEvent ev(String type, long ts) {
        return AdEvent.parse("{\"type\":\"" + type + "\",\"adId\":\"a\",\"ts\":" + ts + "}");
    }

    @Test
    void countsImpressionsClicksAndLatestTimestamp() {
        var agg = new CountAggregate();
        long[] acc = agg.createAccumulator();
        acc = agg.add(ev("impression", 10), acc);
        acc = agg.add(ev("impression", 30), acc);
        acc = agg.add(ev("click", 20), acc);
        assertArrayEquals(new long[] {2, 1, 30}, agg.getResult(acc));
        assertArrayEquals(new long[] {4, 2, 30}, agg.merge(acc, new long[] {2, 1, 25}));
    }

    @Test
    void watermarkFallsBackToWallClockWhenTrafficStops() {
        long[] wm = {Long.MIN_VALUE};
        var out = new org.apache.flink.api.common.eventtime.WatermarkOutput() {
            public void emitWatermark(org.apache.flink.api.common.eventtime.Watermark w) { wm[0] = w.getTimestamp(); }
            public void markIdle() {}
            public void markActive() {}
        };
        long old = System.currentTimeMillis() - 60_000; // last event a minute ago, nothing since
        var gen = new WallClockFallbackWatermarks(2_000, 5_000);
        gen.onEvent(ev("impression", old), old, out);
        gen.onPeriodicEmit(out);
        org.junit.jupiter.api.Assertions.assertTrue(wm[0] >= System.currentTimeMillis() - 6_000);

        var pure = new WallClockFallbackWatermarks(2_000, 0); // fallback off: pure event time
        pure.onEvent(ev("impression", old), old, out);
        pure.onPeriodicEmit(out);
        org.junit.jupiter.api.Assertions.assertEquals(old - 2_001, wm[0]);
    }

    @Test
    void parserRejectsMalformedAndUnknownEvents() {
        assertNotNull(AdEvent.parse("{\"type\":\"click\",\"adId\":\"a\",\"ts\":5,\"extra\":1}"));
        assertNull(AdEvent.parse("not json"));
        assertNull(AdEvent.parse("{\"type\":\"view\",\"adId\":\"a\",\"ts\":5}"));
        assertNull(AdEvent.parse("{\"type\":\"click\",\"ts\":5}"));
        assertNull(AdEvent.parse("{\"type\":\"click\",\"adId\":\"a\"}"));
        assertNull(AdEvent.parse(null));
    }
}
