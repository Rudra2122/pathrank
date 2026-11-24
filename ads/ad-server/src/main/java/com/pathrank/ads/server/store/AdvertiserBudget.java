package com.pathrank.ads.server.store;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import java.time.LocalDate;

/**
 * One advertiser's spend for one UTC day. {@code @Version} is the whole concurrency story: two ad
 * servers that read the same row and both try to add a click get one success and one
 * {@code OptimisticLockException}, and the loser re-reads and tries again.
 */
@Entity
@Table(name = "advertiser_budget")
public class AdvertiserBudget {
    @Id
    @Column(name = "advertiser_id", length = 64)
    private String advertiserId;

    @Column(name = "budget_day", nullable = false)
    private LocalDate budgetDay;

    @Column(name = "spent_micros", nullable = false)
    private long spentMicros;

    @Version
    private Long version;   // null = not persisted yet, so Spring Data inserts instead of merging

    protected AdvertiserBudget() {}

    public AdvertiserBudget(String advertiserId, LocalDate budgetDay) {
        this.advertiserId = advertiserId;
        this.budgetDay = budgetDay;
    }

    public String getAdvertiserId() { return advertiserId; }
    public LocalDate getBudgetDay() { return budgetDay; }
    public long getSpentMicros() { return spentMicros; }
    public Long getVersion() { return version; }

    /** A new UTC day starts from zero. */
    public void rollTo(LocalDate day) {
        if (!day.equals(budgetDay)) {
            budgetDay = day;
            spentMicros = 0;
        }
    }

    public void add(long micros) { spentMicros += micros; }
}
