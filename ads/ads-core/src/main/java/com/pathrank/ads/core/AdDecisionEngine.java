package com.pathrank.ads.core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * One ad request end to end: filter campaigns (budget, frequency cap, job already shown
 * organically), run the GSP auction for the sponsored positions, then keep a winner only if the
 * slot-blending rule says its expected revenue beats the relevance it costs the organic list.
 */
public final class AdDecisionEngine {

    public record Placement(int position, Campaign campaign, double pctr, long priceMicros,
                            double expectedRevenue, double relevanceLoss) {}

    public record Result(List<Placement> placements, int candidates, int filteredBudget,
                         int filteredFrequency, int filteredDuplicate, int rejectedByBlender) {}

    private final GspAuction auction;
    private final BudgetLedger budget;
    private final FrequencyCapper frequency;
    private final CtrSource ctr;
    private final double valuePerApplyUsd;
    private final double alpha;

    public AdDecisionEngine(GspAuction auction, BudgetLedger budget, FrequencyCapper frequency, CtrSource ctr,
                            double valuePerApplyUsd, double alpha) {
        this.auction = auction;
        this.budget = budget;
        this.frequency = frequency;
        this.ctr = ctr;
        this.valuePerApplyUsd = valuePerApplyUsd;
        this.alpha = alpha;
    }

    /**
     * @param positions sponsored positions in the final list (0-based), lowest first, e.g. [2, 5]
     */
    public Result decide(long userId, List<OrganicItem> organic, Collection<Campaign> campaigns, int[] positions) {
        Set<Long> organicJobs = new HashSet<>();
        for (OrganicItem o : organic) organicJobs.add(o.jobId());

        int fBudget = 0, fFreq = 0, fDup = 0;
        List<GspAuction.Candidate> cands = new ArrayList<>();
        for (Campaign c : campaigns) {
            if (organicJobs.contains(c.jobId())) { fDup++; continue; }
            if (!budget.canAfford(c.advertiserId(), c.dailyBudgetMicros(), c.bidMicros())) { fBudget++; continue; }
            if (!frequency.allowed(userId, c.adId())) { fFreq++; continue; }
            cands.add(new GspAuction.Candidate(c, ctr.pctr(c.adId())));
        }

        List<GspAuction.Winner> winners = auction.run(cands, positions.length);

        double[] scores = new double[organic.size()];
        for (int i = 0; i < scores.length; i++) scores[i] = organic.get(i).score();

        List<Placement> placements = new ArrayList<>();
        int rejected = 0;
        for (int i = 0; i < winners.size(); i++) {
            GspAuction.Winner w = winners.get(i);
            int pos = Math.min(positions[i], scores.length);
            double priceUsd = w.priceMicros() / 1_000_000.0;
            SlotBlender.Decision d = SlotBlender.decide(scores, w.pctr(), priceUsd, pos, valuePerApplyUsd, alpha);
            if (!d.insert()) { rejected++; continue; }
            frequency.record(userId, w.campaign().adId());
            placements.add(new Placement(pos, w.campaign(), w.pctr(), w.priceMicros(), d.expectedRevenue(), d.relevanceLoss()));
        }
        return new Result(placements, cands.size(), fBudget, fFreq, fDup, rejected);
    }
}
