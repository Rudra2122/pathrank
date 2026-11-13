package com.pathrank.ads.server;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Served impressions waiting for a possible click. A click is charged only once per impression
 * (the {@code clicked} flag flips with compare-and-set) and only within the TTL.
 */
public final class ImpressionCache {
    public record Impression(String impressionId, String adId, String advertiserId, long userId, long jobId,
                             int position, long priceMicros, long servedAt, AtomicBoolean clicked) {}

    private final Map<String, Impression> byId = new ConcurrentHashMap<>();

    public void put(Impression i) {
        byId.put(i.impressionId(), i);
    }

    public Impression get(String impressionId) {
        return byId.get(impressionId);
    }

    public int evictOlderThan(long cutoffMillis) {
        int before = byId.size();
        byId.values().removeIf(i -> i.servedAt() < cutoffMillis);
        return before - byId.size();
    }

    public int size() {
        return byId.size();
    }
}
