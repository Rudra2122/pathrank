package com.pathrank.ads.server.store;

import com.pathrank.ads.core.Campaign;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Table;

/** A campaign row, so the advertiser console can list "my campaigns" with one indexed query. */
@Entity
@Table(name = "campaign", indexes = @Index(name = "idx_campaign_advertiser", columnList = "advertiser_id"))
public class CampaignEntity {
    @Id
    @Column(name = "ad_id", length = 64)
    private String adId;

    @Column(name = "advertiser_id", nullable = false, length = 64)
    private String advertiserId;

    @Column(name = "job_id", nullable = false)
    private long jobId;

    @Column(name = "bid_micros", nullable = false)
    private long bidMicros;

    @Column(name = "daily_budget_micros", nullable = false)
    private long dailyBudgetMicros;

    protected CampaignEntity() {}

    public static CampaignEntity of(Campaign c) {
        CampaignEntity e = new CampaignEntity();
        e.adId = c.adId();
        e.advertiserId = c.advertiserId();
        e.jobId = c.jobId();
        e.bidMicros = c.bidMicros();
        e.dailyBudgetMicros = c.dailyBudgetMicros();
        return e;
    }

    public String getAdId() { return adId; }
    public String getAdvertiserId() { return advertiserId; }
    public long getJobId() { return jobId; }
    public long getBidMicros() { return bidMicros; }
    public long getDailyBudgetMicros() { return dailyBudgetMicros; }
}
