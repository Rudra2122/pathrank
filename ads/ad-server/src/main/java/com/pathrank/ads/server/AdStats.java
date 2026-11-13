package com.pathrank.ads.server;

import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/** Server-side counts per ad (what this instance served and charged), for GET /ads/stats. */
public final class AdStats {
    public record Row(long impressions, long clicks, long chargedClicks, long spendMicros) {}

    private static final class Acc {
        final LongAdder impressions = new LongAdder(), clicks = new LongAdder(),
                charged = new LongAdder(), spend = new LongAdder();
    }

    private final Map<String, Acc> byAd = new ConcurrentHashMap<>();
    private final LongAdder requests = new LongAdder(), filledRequests = new LongAdder(),
            budgetFiltered = new LongAdder(), frequencyFiltered = new LongAdder(), blenderRejected = new LongAdder();

    public void request(boolean filled, int budget, int freq, int rejected) {
        requests.increment();
        if (filled) filledRequests.increment();
        budgetFiltered.add(budget);
        frequencyFiltered.add(freq);
        blenderRejected.add(rejected);
    }

    public void impression(String adId) {
        acc(adId).impressions.increment();
    }

    public void click(String adId, boolean charged, long priceMicros) {
        Acc a = acc(adId);
        a.clicks.increment();
        if (charged) {
            a.charged.increment();
            a.spend.add(priceMicros);
        }
    }

    public Map<String, Row> perAd() {
        Map<String, Row> out = new TreeMap<>();
        byAd.forEach((k, a) -> out.put(k, new Row(a.impressions.sum(), a.clicks.sum(), a.charged.sum(), a.spend.sum())));
        return out;
    }

    public Map<String, Long> totals() {
        Map<String, Long> t = new TreeMap<>();
        t.put("requests", requests.sum());
        t.put("filled_requests", filledRequests.sum());
        t.put("budget_filtered_candidates", budgetFiltered.sum());
        t.put("frequency_filtered_candidates", frequencyFiltered.sum());
        t.put("blender_rejected_winners", blenderRejected.sum());
        long imp = 0, clk = 0, ch = 0, sp = 0;
        for (Row r : perAd().values()) { imp += r.impressions(); clk += r.clicks(); ch += r.chargedClicks(); sp += r.spendMicros(); }
        t.put("impressions", imp);
        t.put("clicks", clk);
        t.put("charged_clicks", ch);
        t.put("spend_micros", sp);
        return t;
    }

    private Acc acc(String adId) {
        return byAd.computeIfAbsent(adId, k -> new Acc());
    }
}
