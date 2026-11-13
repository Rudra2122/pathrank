package com.pathrank.ads.server;

import com.pathrank.ads.core.CtrEstimate;
import com.pathrank.ads.core.CtrSource;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.core.RedisCallback;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * pCTR per ad from the counts the Flink job writes to Redis ({@code ctr:<adId>} hash with
 * {@code impressions} and {@code clicks}).
 *
 * <p>The auction never calls Redis: a background refresh pulls every ad's counts in one pipelined
 * round trip every {@code ads.ctr-refresh-ms} and swaps them into a local map. The hot path is a
 * map lookup, and CTRs are at most one refresh interval staler than Redis. If Redis is down, the
 * last good values stay in place (new ads fall back to the Beta prior).
 */
@Component
public class RedisCtrSource implements CtrSource {
    private static final Logger log = LoggerFactory.getLogger(RedisCtrSource.class);
    private static final byte[] IMPR = "impressions".getBytes(StandardCharsets.UTF_8);
    private static final byte[] CLICKS = "clicks".getBytes(StandardCharsets.UTF_8);

    public record Counts(long impressions, long clicks) {}

    private final StringRedisTemplate redis;
    private final CampaignStore campaigns;
    private volatile Map<String, Counts> counts = Map.of();
    private volatile long lastRefreshMillis = 0;

    public RedisCtrSource(StringRedisTemplate redis, CampaignStore campaigns) {
        this.redis = redis;
        this.campaigns = campaigns;
    }

    @Override
    public double pctr(String adId) {
        Counts c = counts.get(adId);
        return c == null ? CtrEstimate.smoothed(0, 0) : CtrEstimate.smoothed(c.clicks(), c.impressions());
    }

    public Counts counts(String adId) {
        return counts.getOrDefault(adId, new Counts(0, 0));
    }

    /** {impressions, clicks} for Thompson sampling, or null if Flink hasn't counted this ad yet. */
    public long[] rawCounts(String adId) {
        Counts c = counts.get(adId);
        return c == null ? null : new long[] {c.impressions(), c.clicks()};
    }

    public long lastRefreshMillis() {
        return lastRefreshMillis;
    }

    @Scheduled(fixedDelayString = "${ads.ctr-refresh-ms:500}")
    public void refresh() {
        List<String> ids = new ArrayList<>();
        campaigns.all().forEach(c -> ids.add(c.adId()));
        try {
            List<Object> rows = redis.executePipelined((RedisCallback<Object>) (RedisConnection conn) -> {
                for (String id : ids) {
                    conn.hashCommands().hMGet(("ctr:" + id).getBytes(StandardCharsets.UTF_8), IMPR, CLICKS);
                }
                return null;
            });
            Map<String, Counts> next = new ConcurrentHashMap<>();
            for (int i = 0; i < ids.size(); i++) {
                if (rows.get(i) instanceof List<?> vals && vals.size() == 2) {
                    long imp = parse(vals.get(0)), clk = parse(vals.get(1));
                    if (imp > 0 || clk > 0) next.put(ids.get(i), new Counts(imp, clk));
                }
            }
            counts = next;
            lastRefreshMillis = System.currentTimeMillis();
        } catch (RuntimeException e) {
            log.warn("CTR refresh from Redis failed, keeping previous values: {}", e.toString());
        }
    }

    private static long parse(Object v) {
        if (v == null) return 0;
        try {
            return Long.parseLong(v.toString());
        } catch (NumberFormatException e) {
            return 0;
        }
    }
}
