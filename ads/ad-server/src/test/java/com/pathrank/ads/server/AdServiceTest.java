package com.pathrank.ads.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import com.pathrank.ads.core.AdDecisionEngine;
import com.pathrank.ads.core.BudgetLedger;
import com.pathrank.ads.core.Campaign;
import com.pathrank.ads.core.FrequencyCapper;
import com.pathrank.ads.core.GspAuction;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

/** AdService with the real ads-core logic and a mocked Kafka publisher. */
class AdServiceTest {
    private final AtomicLong now = new AtomicLong(1_000_000);
    private AdEventPublisher events;
    private BudgetLedger budget;
    private AdService service;

    private final List<Dto.Organic> organic = List.of(new Dto.Organic(1, 0.05), new Dto.Organic(2, 0.04),
            new Dto.Organic(3, 0.03), new Dto.Organic(4, 0.02), new Dto.Organic(5, 0.01), new Dto.Organic(6, 0.01));

    @BeforeEach
    void setUp() {
        events = mock(AdEventPublisher.class);
        budget = new BudgetLedger();
        var campaigns = new CampaignStore(List.of(
                new Campaign("ad-1", "adv-1", 9001, 2_000_000, 100_000_000),
                new Campaign("ad-2", "adv-2", 9002, 1_500_000, 2_000_000)));
        Map<String, Double> ctr = Map.of("ad-1", 0.05, "ad-2", 0.04);
        var engine = new AdDecisionEngine(new GspAuction(50_000, 10_000), budget,
                new FrequencyCapper(3, 3_600_000, now::get), id -> ctr.getOrDefault(id, 0.02), 1.0, 1.0);
        service = new AdService(engine, campaigns, budget, new ImpressionCache(), events, new AdStats(),
                new int[] {2, 5}, 600_000, now::get, new SimpleMeterRegistry());
    }

    @Test
    void auctionBlendsSponsoredJobsAtTheirPositionsAndLogsImpressions() {
        var r = service.auction(new Dto.AuctionRequest(42L, organic));
        assertEquals(2, r.sponsoredCount());
        assertEquals(8, r.items().size());
        assertTrue(r.items().get(2).sponsored());
        assertEquals("ad-1", r.items().get(2).adId());
        assertTrue(r.items().get(5).sponsored());
        for (int i = 0; i < r.items().size(); i++) assertEquals(i, r.items().get(i).rank());

        var captor = ArgumentCaptor.forClass(AdEvent.class);
        verify(events, times(2)).publish(captor.capture());
        assertTrue(captor.getAllValues().stream().allMatch(e -> e.type().equals(AdEvent.IMPRESSION)));
    }

    @Test
    void clickIsChargedOnceAndSecondClickIsADuplicate() {
        var r = service.auction(new Dto.AuctionRequest(42L, organic));
        String imp = r.items().get(2).impressionId();

        var first = service.click(imp);
        assertEquals(AdService.ClickStatus.CHARGED, first.status());
        assertEquals(r.items().get(2).priceMicros().longValue(), first.priceMicros());
        assertEquals(first.priceMicros(), budget.spent("adv-1"));

        assertEquals(AdService.ClickStatus.DUPLICATE, service.click(imp).status());
        verify(events, times(3)).publish(any()); // 2 impressions + 1 click
    }

    @Test
    void clickAfterTtlOrOnUnknownIdIsNotCharged() {
        var r = service.auction(new Dto.AuctionRequest(42L, organic));
        now.addAndGet(600_001);
        assertEquals(AdService.ClickStatus.UNKNOWN_OR_EXPIRED, service.click(r.items().get(2).impressionId()).status());
        assertEquals(AdService.ClickStatus.UNKNOWN_OR_EXPIRED, service.click("nope").status());
        assertEquals(0, budget.spent("adv-1"));
    }

    @Test
    void overBudgetClickIsLoggedButNotCharged() {
        var r = service.auction(new Dto.AuctionRequest(42L, organic));
        budget.charge("adv-2", 2_000_000, 1_999_999); // leave less than one click of budget
        var res = service.click(r.items().get(5).impressionId());
        assertEquals(AdService.ClickStatus.BUDGET_EXHAUSTED, res.status());
        var captor = ArgumentCaptor.forClass(AdEvent.class);
        verify(events, times(3)).publish(captor.capture());
        AdEvent click = captor.getAllValues().get(2);
        assertEquals(AdEvent.CLICK, click.type());
        assertFalse(click.charged());
    }
}
