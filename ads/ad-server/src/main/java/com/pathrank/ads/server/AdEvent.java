package com.pathrank.ads.server;

/**
 * One event on the {@code ad-events} Kafka topic, as JSON. The Flink job reads the same fields
 * ({@code com.pathrank.ads.flink.AdEvent}). {@code ts} is epoch millis when the server emitted it.
 */
public record AdEvent(String type, String impressionId, String adId, String advertiserId, long userId,
                      long jobId, int position, long priceMicros, boolean charged, long ts) {
    public static final String IMPRESSION = "impression";
    public static final String CLICK = "click";
}
