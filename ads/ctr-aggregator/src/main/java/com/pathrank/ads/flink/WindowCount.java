package com.pathrank.ads.flink;

/** Impressions and clicks for one ad in one event-time window. */
public class WindowCount {
    public String adId;
    public long windowStart;
    public long windowEnd;
    public long impressions;
    public long clicks;
    /** Latest event timestamp in the window; freshness = time written to Redis - this. */
    public long maxEventTs;

    public WindowCount() {}

    public WindowCount(String adId, long windowStart, long windowEnd, long impressions, long clicks, long maxEventTs) {
        this.adId = adId;
        this.windowStart = windowStart;
        this.windowEnd = windowEnd;
        this.impressions = impressions;
        this.clicks = clicks;
        this.maxEventTs = maxEventTs;
    }

    @Override
    public String toString() {
        return adId + "[" + windowStart + "," + windowEnd + ") imp=" + impressions + " clk=" + clicks;
    }
}
