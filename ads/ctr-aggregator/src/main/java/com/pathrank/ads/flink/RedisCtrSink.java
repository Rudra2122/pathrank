package com.pathrank.ads.flink;

import java.util.List;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import redis.clients.jedis.JedisPooled;

/**
 * Adds each window's counts to {@code ctr:<adId>} (the hash the ad server reads).
 *
 * <p>Idempotent per window: a Lua script first sets a marker key {@code ctrw:<adId>:<windowStart>}
 * with NX, and only increments if the marker was new. If Flink restarts from a checkpoint and
 * re-emits a window it already wrote, the second write is a no-op instead of double counting.
 * Event-time windows with no allowed lateness fire once with deterministic counts, which is what
 * makes this safe.
 *
 * <p>Also pushes a freshness sample (now - latest event time in the window) to
 * {@code ads:freshness_ms}, capped at the last 200K samples.
 */
public class RedisCtrSink extends RichSinkFunction<WindowCount> {
    static final String SCRIPT =
            "if redis.call('SET', KEYS[2], '1', 'NX', 'EX', ARGV[3]) then "
                    + "redis.call('HINCRBY', KEYS[1], 'impressions', ARGV[1]); "
                    + "redis.call('HINCRBY', KEYS[1], 'clicks', ARGV[2]); "
                    + "return 1 else return 0 end";
    static final String FRESHNESS_KEY = "ads:freshness_ms";
    static final String DUPLICATE_WINDOWS_KEY = "ads:duplicate_windows";

    private final String host;
    private final int port;
    private transient JedisPooled jedis;

    public RedisCtrSink(String host, int port) {
        this.host = host;
        this.port = port;
    }

    @Override
    public void open(OpenContext ctx) {
        jedis = new JedisPooled(host, port);
    }

    @Override
    public void invoke(WindowCount w, Context ctx) {
        Object applied = jedis.eval(SCRIPT,
                List.of("ctr:" + w.adId, "ctrw:" + w.adId + ":" + w.windowStart),
                List.of(Long.toString(w.impressions), Long.toString(w.clicks), "86400"));
        if (applied instanceof Long l && l == 0L) {
            jedis.incr(DUPLICATE_WINDOWS_KEY);
            return;
        }
        long freshness = System.currentTimeMillis() - w.maxEventTs;
        jedis.lpush(FRESHNESS_KEY, Long.toString(freshness));
        jedis.ltrim(FRESHNESS_KEY, 0, 199_999);
    }

    @Override
    public void close() {
        if (jedis != null) jedis.close();
    }
}
