package com.pathrank.ads.server;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pathrank.ads.core.Campaign;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Component;

/** Campaigns loaded once at startup from the JSON that sim/make_campaigns.py writes. */
@Component
public class CampaignStore {
    private static final Logger log = LoggerFactory.getLogger(CampaignStore.class);
    private final Map<String, Campaign> byId;

    @Autowired
    public CampaignStore(AdsProperties props, ResourceLoader loader, ObjectMapper mapper) throws IOException {
        try (InputStream in = loader.getResource(props.campaignsPath()).getInputStream()) {
            List<Campaign> list = mapper.readValue(in, new TypeReference<List<Campaign>>() {});
            Map<String, Campaign> m = new LinkedHashMap<>();
            for (Campaign c : list) {
                if (m.put(c.adId(), c) != null) throw new IllegalStateException("duplicate adId " + c.adId());
            }
            this.byId = Collections.unmodifiableMap(m);
        }
        log.info("loaded {} campaigns from {}", byId.size(), props.campaignsPath());
    }

    public CampaignStore(List<Campaign> campaigns) {
        Map<String, Campaign> m = new LinkedHashMap<>();
        campaigns.forEach(c -> m.put(c.adId(), c));
        this.byId = Collections.unmodifiableMap(m);
    }

    public Collection<Campaign> all() {
        return byId.values();
    }

    public Campaign get(String adId) {
        return byId.get(adId);
    }

    public int size() {
        return byId.size();
    }
}
