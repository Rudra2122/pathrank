package com.pathrank.ads.flink;

import org.apache.flink.api.common.functions.AggregateFunction;

/**
 * Incremental count per window: the accumulator is {impressions, clicks, maxEventTs}, so Flink
 * keeps three longs per ad per open window instead of buffering every event.
 */
public class CountAggregate implements AggregateFunction<AdEvent, long[], long[]> {
    @Override
    public long[] createAccumulator() {
        return new long[] {0, 0, 0};
    }

    @Override
    public long[] add(AdEvent e, long[] acc) {
        if (e.isImpression()) acc[0]++;
        else if (e.isClick()) acc[1]++;
        acc[2] = Math.max(acc[2], e.ts);
        return acc;
    }

    @Override
    public long[] getResult(long[] acc) {
        return acc;
    }

    @Override
    public long[] merge(long[] a, long[] b) {
        return new long[] {a[0] + b[0], a[1] + b[1], Math.max(a[2], b[2])};
    }
}
