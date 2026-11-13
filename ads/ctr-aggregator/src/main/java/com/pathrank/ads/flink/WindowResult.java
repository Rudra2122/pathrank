package com.pathrank.ads.flink;

import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

/** Attaches the ad id and window bounds to the pre-aggregated counts. */
public class WindowResult extends ProcessWindowFunction<long[], WindowCount, String, TimeWindow> {
    @Override
    public void process(String adId, Context ctx, Iterable<long[]> counts, Collector<WindowCount> out) {
        long[] c = counts.iterator().next();
        out.collect(new WindowCount(adId, ctx.window().getStart(), ctx.window().getEnd(), c[0], c[1], c[2]));
    }
}
