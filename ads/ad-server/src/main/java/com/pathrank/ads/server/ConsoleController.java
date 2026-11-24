package com.pathrank.ads.server;

import com.pathrank.ads.core.BudgetStore;
import com.pathrank.ads.server.security.Caller;
import com.pathrank.ads.server.store.CampaignEntity;
import com.pathrank.ads.server.store.CampaignRepository;
import com.pathrank.ads.server.store.JpaBudgetStore;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

/**
 * Read API behind the advertiser console. Tenant isolation is enforced here, on every row: an
 * advertiser token only ever sees campaigns whose advertiserId matches its own claim, and asking
 * for someone else's campaign by id is a 403, not an empty result.
 */
@RestController
@RequestMapping("/api")
public class ConsoleController {
    public record CampaignView(String adId, String advertiserId, long jobId, long bidMicros, long dailyBudgetMicros) {
        static CampaignView of(CampaignEntity e) {
            return new CampaignView(e.getAdId(), e.getAdvertiserId(), e.getJobId(), e.getBidMicros(), e.getDailyBudgetMicros());
        }
    }

    public record CampaignStats(String adId, long impressions, long clicks, long chargedClicks, long spendMicros,
                                double pctr) {}

    public record Me(String username, boolean admin, String advertiserId, Long spentTodayMicros) {}

    private final CampaignRepository campaigns;
    private final AdStats stats;
    private final RedisCtrSource ctr;
    private final BudgetStore budget;

    public ConsoleController(CampaignRepository campaigns, AdStats stats, RedisCtrSource ctr, BudgetStore budget) {
        this.campaigns = campaigns;
        this.stats = stats;
        this.ctr = ctr;
        this.budget = budget;
    }

    @GetMapping("/me")
    public Me me(Authentication auth) {
        Caller c = Caller.from(auth);
        Long spent = c.advertiserId() == null ? null : spentOf(c.advertiserId());
        return new Me(c.username(), c.admin(), c.advertiserId(), spent);
    }

    @GetMapping("/campaigns")
    public List<CampaignView> list(Authentication auth) {
        Caller c = Caller.from(auth);
        List<CampaignEntity> rows = c.admin() ? campaigns.findAllByOrderByAdId()
                : campaigns.findByAdvertiserIdOrderByAdId(c.advertiserId());
        List<CampaignView> out = new ArrayList<>(rows.size());
        for (CampaignEntity e : rows) out.add(CampaignView.of(e));
        return out;
    }

    @GetMapping("/campaigns/{adId}")
    public CampaignView one(@PathVariable("adId") String adId, Authentication auth) {
        return CampaignView.of(owned(adId, Caller.from(auth)));
    }

    @GetMapping("/campaigns/{adId}/stats")
    public CampaignStats campaignStats(@PathVariable("adId") String adId, Authentication auth) {
        CampaignEntity e = owned(adId, Caller.from(auth));
        AdStats.Row row = stats.perAd().get(e.getAdId());
        return row == null
                ? new CampaignStats(adId, 0, 0, 0, 0, ctr.pctr(adId))
                : new CampaignStats(adId, row.impressions(), row.clicks(), row.chargedClicks(), row.spendMicros(),
                        ctr.pctr(adId));
    }

    @GetMapping("/advertisers/{advertiserId}/spend")
    public Map<String, Object> spend(@PathVariable("advertiserId") String advertiserId, Authentication auth) {
        if (!Caller.from(auth).canSee(advertiserId)) throw new AccessDeniedException("not your advertiser account");
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("advertiserId", advertiserId);
        out.put("spentTodayMicros", spentOf(advertiserId));
        return out;
    }

    private CampaignEntity owned(String adId, Caller c) {
        CampaignEntity e = campaigns.findById(adId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "no campaign " + adId));
        if (!c.canSee(e.getAdvertiserId())) throw new AccessDeniedException("campaign belongs to another advertiser");
        return e;
    }

    private long spentOf(String advertiserId) {
        return budget instanceof JpaBudgetStore j ? j.spentFromDatabase(advertiserId) : budget.spent(advertiserId);
    }
}
