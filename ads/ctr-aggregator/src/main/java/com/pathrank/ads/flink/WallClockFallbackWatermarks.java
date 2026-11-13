package com.pathrank.ads.flink;

import org.apache.flink.api.common.eventtime.Watermark;
import org.apache.flink.api.common.eventtime.WatermarkGenerator;
import org.apache.flink.api.common.eventtime.WatermarkOutput;

/**
 * Bounded-out-of-orderness watermarks, plus a wall-clock floor so windows still close when
 * traffic stops.
 *
 * <p>Plain event-time watermarks only move when new events arrive. If an ad gets no more traffic,
 * its last window never fires, so its clicks never reach Redis (in the first benchmark run, every
 * ad's final window sat there until unrelated load restarted ~30 s later, which put p95 freshness
 * at 30 s). Event timestamps here are the ad server's wall clock at emit time, so the watermark can
 * safely be at least {@code now - fallbackMs}: an event older than that is genuinely late and goes
 * to the late-events side output instead of being silently dropped.
 *
 * <p>{@code fallbackMs = 0} turns the floor off (pure event time), which the tests use because
 * their timestamps are fixed numbers, not wall-clock times.
 */
public class WallClockFallbackWatermarks implements WatermarkGenerator<AdEvent> {
    private final long outOfOrderMs;
    private final long fallbackMs;
    private long maxTs;

    public WallClockFallbackWatermarks(long outOfOrderMs, long fallbackMs) {
        if (fallbackMs > 0 && fallbackMs <= outOfOrderMs) {
            throw new IllegalArgumentException("wall-clock fallback must be larger than the out-of-order bound");
        }
        this.outOfOrderMs = outOfOrderMs;
        this.fallbackMs = fallbackMs;
        this.maxTs = Long.MIN_VALUE + outOfOrderMs + 1;
    }

    @Override
    public void onEvent(AdEvent e, long eventTimestamp, WatermarkOutput out) {
        maxTs = Math.max(maxTs, eventTimestamp);
    }

    @Override
    public void onPeriodicEmit(WatermarkOutput out) {
        long wm = maxTs - outOfOrderMs - 1;
        if (fallbackMs > 0) wm = Math.max(wm, System.currentTimeMillis() - fallbackMs);
        out.emitWatermark(new Watermark(wm));
    }
}
