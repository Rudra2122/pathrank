package com.pathrank.ads.server;

import com.pathrank.ads.core.BudgetStore;
import jakarta.validation.Valid;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/ads")
public class AdController {
    private final AdService service;
    private final AdStats stats;
    private final RedisCtrSource ctr;
    private final CampaignStore campaigns;
    private final BudgetStore budget;

    public AdController(AdService service, AdStats stats, RedisCtrSource ctr, CampaignStore campaigns, BudgetStore budget) {
        this.service = service;
        this.stats = stats;
        this.ctr = ctr;
        this.campaigns = campaigns;
        this.budget = budget;
    }

    /** Organic list in, organic list with sponsored jobs blended in out. */
    @PostMapping("/auction")
    public Dto.AuctionResponse auction(@Valid @RequestBody Dto.AuctionRequest req) {
        return service.auction(req);
    }

    /** 200 charged / 200 not charged (budget gone) / 409 already clicked / 404 unknown or expired. */
    @PostMapping("/click")
    public ResponseEntity<Dto.ClickResponse> click(@Valid @RequestBody Dto.ClickRequest req) {
        AdService.ClickResult r = service.click(req.impressionId());
        Dto.ClickResponse body = new Dto.ClickResponse(r.status().name(), r.status() == AdService.ClickStatus.CHARGED,
                r.priceMicros());
        return switch (r.status()) {
            case CHARGED, BUDGET_EXHAUSTED -> ResponseEntity.ok(body);
            case DUPLICATE -> ResponseEntity.status(HttpStatus.CONFLICT).body(body);
            case UNKNOWN_OR_EXPIRED -> ResponseEntity.status(HttpStatus.NOT_FOUND).body(body);
        };
    }

    /** Totals plus per-ad served/clicked/spend (this instance) and the Flink counts + pCTR it is using. */
    @GetMapping("/stats")
    public Map<String, Object> stats() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("totals", stats.totals());
        out.put("ctr_last_refresh_ms", ctr.lastRefreshMillis());
        Map<String, Object> ads = new LinkedHashMap<>();
        stats.perAd().forEach((adId, row) -> {
            var c = campaigns.get(adId);
            var flink = ctr.counts(adId);
            Map<String, Object> a = new LinkedHashMap<>();
            a.put("served_impressions", row.impressions());
            a.put("clicks", row.clicks());
            a.put("charged_clicks", row.chargedClicks());
            a.put("spend_micros", row.spendMicros());
            a.put("flink_impressions", flink.impressions());
            a.put("flink_clicks", flink.clicks());
            a.put("pctr", ctr.pctr(adId));
            if (c != null) {
                a.put("bid_micros", c.bidMicros());
                a.put("advertiser_spend_micros", budget.spent(c.advertiserId()));
            }
            ads.put(adId, a);
        });
        out.put("ads", ads);
        return out;
    }
}
