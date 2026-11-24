package com.pathrank.ads.server.store;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface CampaignRepository extends JpaRepository<CampaignEntity, String> {
    List<CampaignEntity> findByAdvertiserIdOrderByAdId(String advertiserId);

    List<CampaignEntity> findAllByOrderByAdId();
}
