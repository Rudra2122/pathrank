package com.pathrank.ads.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.CloseableIterator;
import org.junit.jupiter.api.Test;

/** Runs the real windowing pipeline on an embedded Flink mini cluster with a bounded input. */
class CtrWindowJobTest {
    private static String ev(String type, String ad, long ts) {
        return "{\"type\":\"" + type + "\",\"adId\":\"" + ad + "\",\"ts\":" + ts + "}";
    }

    @Test
    void countsPerAdPerTumblingWindowAndDropsMalformed() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);
        var input = env.fromData(
                ev("impression", "a", 1_000), ev("impression", "a", 2_000), ev("click", "a", 4_999),
                ev("impression", "b", 3_000),
                "garbage",
                ev("impression", "a", 6_000), ev("click", "a", 7_000),   // second 5 s window
                ev("click", "b", 9_999));

        List<WindowCount> out = new ArrayList<>();
        try (CloseableIterator<WindowCount> it = CtrAggregatorJob.windows(input, 5_000, 0, 0).executeAndCollect()) {
            it.forEachRemaining(out::add);
        }
        out.sort(Comparator.comparing((WindowCount w) -> w.adId).thenComparingLong(w -> w.windowStart));

        assertEquals(4, out.size(), out.toString());
        assertWindow(out.get(0), "a", 0, 2, 1, 4_999);
        assertWindow(out.get(1), "a", 5_000, 1, 1, 7_000);
        assertWindow(out.get(2), "b", 0, 1, 0, 3_000);
        assertWindow(out.get(3), "b", 5_000, 0, 1, 9_999);
    }

    private static void assertWindow(WindowCount w, String ad, long start, long imp, long clk, long maxTs) {
        assertEquals(ad, w.adId);
        assertEquals(start, w.windowStart);
        assertEquals(start + 5_000, w.windowEnd);
        assertEquals(imp, w.impressions);
        assertEquals(clk, w.clicks);
        assertEquals(maxTs, w.maxEventTs);
    }
}
