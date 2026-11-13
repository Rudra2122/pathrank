package com.pathrank.ads.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

/**
 * Fire-and-forget send to Kafka, keyed by adId so one ad's events stay ordered on one partition
 * (the Flink job keys by adId too). The request thread never waits for the broker ack; failures
 * are counted in {@code ads.events.send_errors} instead of failing the auction.
 */
@Component
public class KafkaAdEventPublisher implements AdEventPublisher {
    private static final Logger log = LoggerFactory.getLogger(KafkaAdEventPublisher.class);
    private final KafkaTemplate<String, String> kafka;
    private final ObjectMapper mapper;
    private final String topic;
    private final Counter errors;

    public KafkaAdEventPublisher(KafkaTemplate<String, String> kafka, ObjectMapper mapper, AdsProperties props,
                                 MeterRegistry registry) {
        this.kafka = kafka;
        this.mapper = mapper;
        this.topic = props.eventsTopic();
        this.errors = Counter.builder("ads.events.send_errors").register(registry);
    }

    @Override
    public void publish(AdEvent event) {
        String json;
        try {
            json = mapper.writeValueAsString(event);
        } catch (JsonProcessingException e) {
            errors.increment();
            log.error("could not serialize {}", event, e);
            return;
        }
        kafka.send(topic, event.adId(), json).whenComplete((r, ex) -> {
            if (ex != null) {
                errors.increment();
                log.warn("ad event send failed: {}", ex.toString());
            }
        });
    }
}
