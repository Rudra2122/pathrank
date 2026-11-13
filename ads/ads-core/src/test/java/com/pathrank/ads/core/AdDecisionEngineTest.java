package com.pathrank.ads.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class AdDecisionEngineTest {
    private final AtomicLong now = new AtomicLong(0);
    private final List<OrganicItem> organic = List.of(
            new OrganicItem(1, 0.05), new OrganicItem(2, 0.04), new OrganicItem(3, 0.03),
            new OrganicItem(4, 0.02), new OrganicItem(5, 0.01), new OrganicItem(6, 0.01));

    private AdDecisionEngine engine(BudgetLedger budget, Map<String, Double> ctr) {
        return new AdDecisionEngine(new GspAuction(10_000, 10_000), budget,
                new FrequencyCapper(1, 3_600_000, now::get), id -> ctr.getOrDefault(id, 0.02), 1.0, 1.0);
    }

    @Test
    void fillsSponsoredPositionsAndSkipsJobsAlreadyShown() {
        var camps = List.of(
                new Campaign("a1", "adv1", 100, 2_000_000, 50_000_000),
                new Campaign("a2", "adv2", 101, 1_500_000, 50_000_000),
                new Campaign("dup", "adv3", 3, 9_000_000, 50_000_000)); // job 3 is already organic
        var r = engine(new BudgetLedger(), Map.of("a1", 0.05, "a2", 0.04)).decide(7, organic, camps, new int[] {2, 5});
        assertEquals(1, r.filteredDuplicate());
        assertEquals(2, r.placements().size());
        assertEquals("a1", r.placements().get(0).campaign().adId());
        assertEquals(2, r.placements().get(0).position());
    }

    @Test
    void frequencyCapHidesAnAdAfterItWasShown() {
        var camps = List.of(new Campaign("a1", "adv1", 100, 2_000_000, 50_000_000));
        var e = engine(new BudgetLedger(), Map.of());
        assertEquals(1, e.decide(7, organic, camps, new int[] {2}).placements().size());
        var second = e.decide(7, organic, camps, new int[] {2});
        assertEquals(0, second.placements().size());
        assertEquals(1, second.filteredFrequency());
    }

    @Test
    void exhaustedBudgetRemovesCampaign() {
        var budget = new BudgetLedger();
        budget.charge("adv1", 1_000_000, 900_000);
        var camps = List.of(new Campaign("a1", "adv1", 100, 200_000, 1_000_000));
        var r = engine(budget, Map.of()).decide(7, organic, camps, new int[] {2});
        assertEquals(1, r.filteredBudget());
        assertTrue(r.placements().isEmpty());
    }
}
