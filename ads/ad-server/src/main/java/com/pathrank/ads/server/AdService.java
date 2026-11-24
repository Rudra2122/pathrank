package com.pathrank.ads.server;

import com.pathrank.ads.core.AdDecisionEngine;
import com.pathrank.ads.core.BudgetStore;
import com.pathrank.ads.core.Campaign;
import com.pathrank.ads.core.OrganicItem;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.LongSupplier;

/** The two things the ad server does: run an auction for a result list, and charge a click. */
public class AdService {
    public enum ClickStatus { CHARGED, BUDGET_EXHAUSTED, DUPLICATE, UNKNOWN_OR_EXPIRED }

    public record ClickResult(ClickStatus status, long priceMicros) {}

    private final AdDecisionEngine engine;
    private final CampaignStore campaigns;
    private final BudgetStore budget;
    private final ImpressionCache impressions;
    private final AdEventPublisher events;
    private final AdStats stats;
    private final int[] positions;
    private final long impressionTtlMs;
    private final LongSupplier clock;
    private final Timer auctionTimer;
    private final Counter impressionCounter, clickCounter;

    public AdService(AdDecisionEngine engine, CampaignStore campaigns, BudgetStore budget, ImpressionCache impressions,
                     AdEventPublisher events, AdStats stats, int[] positions, long impressionTtlMs,
                     LongSupplier clock, MeterRegistry registry) {
        this.engine = engine;
        this.campaigns = campaigns;
        this.budget = budget;
        this.impressions = impressions;
        this.events = events;
        this.stats = stats;
        this.positions = positions;
        this.impressionTtlMs = impressionTtlMs;
        this.clock = clock;
        this.auctionTimer = Timer.builder("ads.auction.latency").publishPercentileHistogram().register(registry);
        this.impressionCounter = Counter.builder("ads.impressions").register(registry);
        this.clickCounter = Counter.builder("ads.clicks").register(registry);
    }

    public Dto.AuctionResponse auction(Dto.AuctionRequest req) {
        return auctionTimer.record(() -> runAuction(req));
    }

    private Dto.AuctionResponse runAuction(Dto.AuctionRequest req) {
        List<OrganicItem> organic = new ArrayList<>(req.organic().size());
        for (Dto.Organic o : req.organic()) organic.add(new OrganicItem(o.jobId(), o.score()));

        AdDecisionEngine.Result r = engine.decide(req.userId(), organic, campaigns.all(), positions);
        stats.request(!r.placements().isEmpty(), r.filteredBudget(), r.filteredFrequency(), r.rejectedByBlender());

        List<Dto.Item> items = new ArrayList<>(organic.size() + r.placements().size());
        for (OrganicItem o : organic) items.add(new Dto.Item(0, o.jobId(), o.score(), false, null, null, null));

        long now = clock.getAsLong();
        // placements come back in ascending position order, so inserting in order lands each one
        // at its final index
        for (AdDecisionEngine.Placement p : r.placements()) {
            Campaign c = p.campaign();
            String impId = UUID.randomUUID().toString();
            int at = Math.min(p.position(), items.size());
            items.add(at, new Dto.Item(0, c.jobId(), p.pctr(), true, c.adId(), impId, p.priceMicros()));
            impressions.put(new ImpressionCache.Impression(impId, c.adId(), c.advertiserId(), req.userId(), c.jobId(),
                    at, p.priceMicros(), now, new AtomicBoolean(false)));
            events.publish(new AdEvent(AdEvent.IMPRESSION, impId, c.adId(), c.advertiserId(), req.userId(), c.jobId(),
                    at, p.priceMicros(), false, now));
            stats.impression(c.adId());
            impressionCounter.increment();
        }

        List<Dto.Item> ranked = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) {
            Dto.Item it = items.get(i);
            ranked.add(new Dto.Item(i, it.jobId(), it.score(), it.sponsored(), it.adId(), it.impressionId(), it.priceMicros()));
        }
        return new Dto.AuctionResponse(UUID.randomUUID().toString(), ranked, r.placements().size(), r.candidates());
    }

    public ClickResult click(String impressionId) {
        ImpressionCache.Impression imp = impressions.get(impressionId);
        long now = clock.getAsLong();
        if (imp == null || now - imp.servedAt() > impressionTtlMs) {
            return new ClickResult(ClickStatus.UNKNOWN_OR_EXPIRED, 0);
        }
        if (!imp.clicked().compareAndSet(false, true)) {
            return new ClickResult(ClickStatus.DUPLICATE, 0);
        }
        Campaign c = campaigns.get(imp.adId());
        boolean charged = c != null && budget.charge(c.advertiserId(), c.dailyBudgetMicros(), imp.priceMicros());
        long price = charged ? imp.priceMicros() : 0;
        // the click is logged either way: an over-budget click still counts toward the ad's CTR
        events.publish(new AdEvent(AdEvent.CLICK, imp.impressionId(), imp.adId(), imp.advertiserId(), imp.userId(),
                imp.jobId(), imp.position(), price, charged, now));
        stats.click(imp.adId(), charged, price);
        clickCounter.increment();
        return new ClickResult(charged ? ClickStatus.CHARGED : ClickStatus.BUDGET_EXHAUSTED, price);
    }
}
