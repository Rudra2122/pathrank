package com.pathrank.ads.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.actuate.observability.AutoConfigureObservability;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.kafka.KafkaContainer;

/**
 * Full service against real Kafka and Redis in Docker (Testcontainers): CTRs written to Redis the
 * way the Flink job writes them change who wins, and impression/click events land on Kafka.
 */
@Testcontainers
@AutoConfigureObservability
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {"ads.campaigns-path=classpath:campaigns-test.json", "ads.ctr-refresh-ms=100",
                "ads.exploration=none"})
class AdServerIntegrationTest {

    @Container
    static final KafkaContainer KAFKA = new KafkaContainer("apache/kafka:3.8.1");

    @Container
    @SuppressWarnings("resource")
    static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7").withExposedPorts(6379);

    @DynamicPropertySource
    static void props(DynamicPropertyRegistry r) {
        r.add("spring.kafka.bootstrap-servers", KAFKA::getBootstrapServers);
        r.add("spring.data.redis.host", REDIS::getHost);
        r.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
    }

    @Autowired TestRestTemplate http;
    @Autowired StringRedisTemplate redis;
    @Autowired ObjectMapper mapper;
    @Autowired RedisCtrSource ctr;

    private Map<String, Object> request(long userId) {
        List<Map<String, Object>> organic = new ArrayList<>();
        for (int j = 1; j <= 6; j++) organic.add(Map.of("jobId", j, "score", 0.06 - j * 0.008));
        return Map.of("userId", userId, "organic", organic);
    }

    @Test
    void auctionUsesRedisCtrAndClickFlowsToKafka() throws Exception {
        // ad-2 bids less than ad-1 but has a much better observed CTR, so it should rank first
        redis.opsForHash().putAll("ctr:ad-1", Map.of("impressions", "10000", "clicks", "100"));  // ~1%
        redis.opsForHash().putAll("ctr:ad-2", Map.of("impressions", "10000", "clicks", "800"));  // ~8%
        ctr.refresh();
        assertTrue(ctr.pctr("ad-2") > 0.07);

        var res = http.postForEntity("/ads/auction", request(7), String.class);
        assertEquals(HttpStatus.OK, res.getStatusCode());
        JsonNode body = mapper.readTree(res.getBody());
        JsonNode first = body.get("items").get(2);
        assertTrue(first.get("sponsored").asBoolean());
        assertEquals("ad-2", first.get("adId").asText());

        String imp = first.get("impressionId").asText();
        var click = http.postForEntity("/ads/click", Map.of("impressionId", imp), String.class);
        assertEquals(HttpStatus.OK, click.getStatusCode());
        assertEquals(HttpStatus.CONFLICT,
                http.postForEntity("/ads/click", Map.of("impressionId", imp), String.class).getStatusCode());
        assertEquals(HttpStatus.NOT_FOUND,
                http.postForEntity("/ads/click", Map.of("impressionId", "missing"), String.class).getStatusCode());

        // other tests in this class also publish, so look for this test's events by id
        List<JsonNode> seen = readEventsUntil(imp);
        assertTrue(seen.stream().anyMatch(e -> e.get("type").asText().equals("impression")
                && e.get("impressionId").asText().equals(imp)));
        JsonNode clickEvent = seen.stream().filter(e -> e.get("type").asText().equals("click")
                && e.get("impressionId").asText().equals(imp)).findFirst().orElseThrow();
        assertEquals("ad-2", clickEvent.get("adId").asText());
        assertTrue(clickEvent.get("charged").asBoolean());
    }

    @Test
    void invalidRequestIsRejected() {
        var res = http.postForEntity("/ads/auction", Map.of("userId", 1, "organic", List.of()), String.class);
        assertEquals(HttpStatus.BAD_REQUEST, res.getStatusCode());
    }

    @Test
    void prometheusEndpointExposesAuctionLatency() {
        http.postForEntity("/ads/auction", request(8), String.class);
        String metrics = http.getForObject("/actuator/prometheus", String.class);
        assertNotNull(metrics);
        assertTrue(metrics.contains("ads_auction_latency_seconds"));
    }

    private List<JsonNode> readEventsUntil(String clickedImpressionId) throws Exception {
        Properties p = new Properties();
        p.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, KAFKA.getBootstrapServers());
        p.put(ConsumerConfig.GROUP_ID_CONFIG, "it-" + System.nanoTime());
        p.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        p.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        p.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        List<JsonNode> out = new ArrayList<>();
        try (KafkaConsumer<String, String> c = new KafkaConsumer<>(p)) {
            c.subscribe(List.of("ad-events"));
            long deadline = System.currentTimeMillis() + 20_000;
            while (System.currentTimeMillis() < deadline && out.stream().noneMatch(e ->
                    e.get("type").asText().equals("click") && e.get("impressionId").asText().equals(clickedImpressionId))) {
                for (ConsumerRecord<String, String> r : c.poll(Duration.ofMillis(500))) out.add(mapper.readTree(r.value()));
            }
        }
        return out;
    }
}
