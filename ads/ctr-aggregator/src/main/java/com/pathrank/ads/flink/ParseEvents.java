package com.pathrank.ads.flink;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.util.Collector;

/** JSON string -> AdEvent; malformed records emit nothing (a Flink map can't return null). */
public class ParseEvents implements FlatMapFunction<String, AdEvent> {
    @Override
    public void flatMap(String json, Collector<AdEvent> out) {
        AdEvent e = AdEvent.parse(json);
        if (e != null) out.collect(e);
    }
}
