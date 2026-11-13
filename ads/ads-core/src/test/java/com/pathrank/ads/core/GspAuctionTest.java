package com.pathrank.ads.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class GspAuctionTest {
    private static Campaign c(String id, long bidMicros) {
        return new Campaign(id, "adv-" + id, id.hashCode(), bidMicros, 100_000_000L);
    }

    @Test
    void ranksByBidTimesPctrNotBidAlone() {
        // A bids more but nobody clicks it; B wins on expected value
        var a = new GspAuction.Candidate(c("A", 2_000_000), 0.01); // 20,000
        var b = new GspAuction.Candidate(c("B", 1_000_000), 0.05); // 50,000
        var w = new GspAuction(10_000, 10_000).run(List.of(a, b), 1);
        assertEquals("B", w.get(0).campaign().adId());
    }

    @Test
    void winnerPaysNextScoreOverOwnPctrPlusIncrement() {
        var a = new GspAuction.Candidate(c("A", 1_000_000), 0.05); // 50,000
        var b = new GspAuction.Candidate(c("B", 800_000), 0.04);   // 32,000
        var w = new GspAuction(10_000, 10_000).run(List.of(a, b), 1);
        // 32,000 / 0.05 = 640,000 + 10,000 increment
        assertEquals(650_000, w.get(0).priceMicros());
    }

    @Test
    void priceNeverExceedsBidAndLastWinnerPaysReserve() {
        var a = new GspAuction.Candidate(c("A", 500_000), 0.02);
        var b = new GspAuction.Candidate(c("B", 499_000), 0.02);
        var w = new GspAuction(50_000, 10_000).run(List.of(a, b), 2);
        assertEquals(500_000, w.get(0).priceMicros()); // 499,000 + 10,000 capped at the 500,000 bid
        assertEquals(50_000, w.get(1).priceMicros());  // nobody below B, so B pays the reserve
        for (var x : w) assertTrue(x.priceMicros() <= x.campaign().bidMicros());
    }

    @Test
    void bidsBelowReserveAndZeroCtrAreDropped() {
        var low = new GspAuction.Candidate(c("LOW", 5_000), 0.5);
        var dead = new GspAuction.Candidate(c("DEAD", 900_000), 0.0);
        var ok = new GspAuction.Candidate(c("OK", 100_000), 0.01);
        var w = new GspAuction(10_000, 10_000).run(List.of(low, dead, ok), 3);
        assertEquals(1, w.size());
        assertEquals("OK", w.get(0).campaign().adId());
    }

    @Test
    void tiesBreakOnAdIdSoResultsAreDeterministic() {
        var x = new GspAuction.Candidate(c("X", 100_000), 0.1);
        var y = new GspAuction.Candidate(c("Y", 100_000), 0.1);
        var w1 = new GspAuction(0, 0).run(List.of(y, x), 1);
        var w2 = new GspAuction(0, 0).run(List.of(x, y), 1);
        assertEquals("X", w1.get(0).campaign().adId());
        assertEquals("X", w2.get(0).campaign().adId());
    }
}
