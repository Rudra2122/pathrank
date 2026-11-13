package com.pathrank.ads.core;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.LongSupplier;

/**
 * Caps how many times one user sees one ad in a rolling window (e.g. 3 per hour), so the same
 * sponsored job doesn't follow a user around. In-memory, so it's per ad-server instance; with more
 * than one instance this state would move to Redis.
 */
public final class FrequencyCapper {
    private final int maxImpressions;
    private final long windowMillis;
    private final LongSupplier clock;
    private final Map<String, Deque<Long>> seen = new ConcurrentHashMap<>();

    public FrequencyCapper(int maxImpressions, long windowMillis, LongSupplier clock) {
        if (maxImpressions <= 0 || windowMillis <= 0) throw new IllegalArgumentException("cap and window must be > 0");
        this.maxImpressions = maxImpressions;
        this.windowMillis = windowMillis;
        this.clock = clock;
    }

    /** True if the user can see this ad now. Does not record anything. */
    public boolean allowed(long userId, String adId) {
        Deque<Long> q = seen.get(key(userId, adId));
        if (q == null) return true;
        long cutoff = clock.getAsLong() - windowMillis;
        synchronized (q) {
            while (!q.isEmpty() && q.peekFirst() <= cutoff) q.pollFirst();
            return q.size() < maxImpressions;
        }
    }

    /** Record one impression. */
    public void record(long userId, String adId) {
        Deque<Long> q = seen.computeIfAbsent(key(userId, adId), k -> new ArrayDeque<>());
        synchronized (q) {
            q.addLast(clock.getAsLong());
        }
    }

    /** Drop entries whose whole history is outside the window, so memory doesn't grow with every user ever seen. */
    public int evictExpired() {
        long cutoff = clock.getAsLong() - windowMillis;
        int[] removed = {0};
        seen.entrySet().removeIf(e -> {
            Deque<Long> q = e.getValue();
            synchronized (q) {
                boolean stale = q.isEmpty() || q.peekLast() <= cutoff;
                if (stale) removed[0]++;
                return stale;
            }
        });
        return removed[0];
    }

    public int trackedPairs() {
        return seen.size();
    }

    private static String key(long userId, String adId) {
        return userId + ":" + adId;
    }
}
