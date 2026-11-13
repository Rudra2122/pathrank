package com.pathrank.ads.flink;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Flink-side copy of the ad server's event (a POJO, so Flink serializes it without Kryo).
 * {@link #parse} returns null for anything malformed; the job filters those out and counts them.
 */
public class AdEvent {
    private static final ObjectMapper MAPPER =
            new ObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    public String type;
    public String impressionId;
    public String adId;
    public String advertiserId;
    public long userId;
    public long jobId;
    public int position;
    public long priceMicros;
    public boolean charged;
    public long ts;

    public AdEvent() {}

    public boolean isImpression() {
        return "impression".equals(type);
    }

    public boolean isClick() {
        return "click".equals(type);
    }

    public static AdEvent parse(String json) {
        if (json == null) return null;
        try {
            AdEvent e = MAPPER.readValue(json, AdEvent.class);
            if (e.adId == null || e.adId.isEmpty() || e.ts <= 0 || !(e.isImpression() || e.isClick())) return null;
            return e;
        } catch (Exception ex) {
            return null;
        }
    }
}
