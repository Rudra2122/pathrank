package com.pathrank.ads.server;

/** Sends ad events downstream. The Kafka implementation is KafkaAdEventPublisher; tests mock this. */
public interface AdEventPublisher {
    void publish(AdEvent event);
}
