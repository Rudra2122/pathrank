package com.pathrank.ads.flink;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import redis.clients.jedis.JedisPooled;

/** INCR a Redis counter once per element (used for late and malformed events). */
public class CounterSink<T> extends RichSinkFunction<T> {
    private final String host;
    private final int port;
    private final String key;
    private transient JedisPooled jedis;

    public CounterSink(String host, int port, String key) {
        this.host = host;
        this.port = port;
        this.key = key;
    }

    @Override
    public void open(OpenContext ctx) {
        jedis = new JedisPooled(host, port);
    }

    @Override
    public void invoke(T value, Context ctx) {
        jedis.incr(key);
    }

    @Override
    public void close() {
        if (jedis != null) jedis.close();
    }
}
