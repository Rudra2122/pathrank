package com.pathrank.ads.server.store;

import com.pathrank.ads.server.CampaignStore;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

/**
 * Copies the campaigns the auction loaded (campaigns.json) into the database once, so the console
 * API can query them. The auction itself keeps using the in-memory {@link CampaignStore}: it reads
 * every campaign on every request, and a database round trip there would cost more than the
 * whole auction does today.
 */
@Component
public class CampaignSeeder implements ApplicationRunner {
    private static final Logger log = LoggerFactory.getLogger(CampaignSeeder.class);
    private final CampaignStore campaigns;
    private final CampaignRepository repo;

    public CampaignSeeder(CampaignStore campaigns, CampaignRepository repo) {
        this.campaigns = campaigns;
        this.repo = repo;
    }

    @Override
    public void run(ApplicationArguments args) {
        List<CampaignEntity> missing = new ArrayList<>();
        for (var c : campaigns.all()) {
            if (!repo.existsById(c.adId())) missing.add(CampaignEntity.of(c));
        }
        if (missing.isEmpty()) return;
        // two ad servers starting together both try to seed. Postgres reports the loser as a duplicate
        // key; CockroachDB (SERIALIZABLE) as a 40001 restart. Either way: re-check and try again.
        for (int attempt = 1; attempt <= 5; attempt++) {
            try {
                List<CampaignEntity> still = new ArrayList<>();
                for (CampaignEntity e : missing) if (!repo.existsById(e.getAdId())) still.add(e);
                if (!still.isEmpty()) repo.saveAll(still);
                log.info("campaigns seeded ({} written)", still.size());
                return;
            } catch (RuntimeException e) {
                if (!JpaBudgetStore.isRetryable(e)) throw e;
                log.info("campaign seeding raced another instance (attempt {}), retrying", attempt);
                try { Thread.sleep(200L * attempt); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); return; }
            }
        }
        log.warn("gave up seeding campaigns after 5 attempts; another instance has most likely seeded them");
    }
}
