package com.pathrank.ads.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.JwsHeader;
import org.springframework.security.oauth2.jwt.JwtClaimsSet;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtEncoderParameters;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.kafka.KafkaContainer;

/**
 * Every way a request should be turned away, against the real filter chain: no token, garbage token,
 * wrong signature, expired token, wrong issuer, wrong role, and an advertiser reaching for another
 * advertiser's campaign. Plus the allowed calls, so a chain that rejects everything can't pass.
 */
@Testcontainers
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, properties = {
        "ads.campaigns-path=classpath:campaigns-test.json",
        "ads.security.enabled=true",
        "ads.security.users[0].username=admin", "ads.security.users[0].password=admin-pass",
        "ads.security.users[0].role=ADMIN",
        "ads.security.users[1].username=jobboard", "ads.security.users[1].password=jobboard-pass",
        "ads.security.users[1].role=SERVICE",
        "ads.security.users[2].username=alice", "ads.security.users[2].password=alice-pass",
        "ads.security.users[2].role=ADVERTISER", "ads.security.users[2].advertiserId=adv-1",
        "ads.security.users[3].username=bob", "ads.security.users[3].password=bob-pass",
        "ads.security.users[3].role=ADVERTISER", "ads.security.users[3].advertiserId=adv-2"})
class SecurityIntegrationTest {

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
    @Autowired ObjectMapper mapper;
    @Autowired JwtEncoder encoder;

    record Case(String name, HttpMethod method, String path, String token, String body, int expected) {}

    private String login(String user, String pass) throws Exception {
        ResponseEntity<String> r = http.postForEntity("/auth/token",
                json(Map.of("username", user, "password", pass), null), String.class);
        assertEquals(200, r.getStatusCode().value(), "login " + user);
        JsonNode n = mapper.readTree(r.getBody());
        return n.get("accessToken").asText();
    }

    private HttpEntity<String> json(Object body, String token) throws Exception {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        if (token != null) h.setBearerAuth(token);
        return new HttpEntity<>(body == null ? null : (body instanceof String s ? s : mapper.writeValueAsString(body)), h);
    }

    /** A token we sign ourselves, to make expired / wrong-issuer tokens with the real key. */
    private String forged(String issuer, Instant expires, String role) {
        JwtClaimsSet claims = JwtClaimsSet.builder().issuer(issuer).subject("mallory").issuedAt(expires.minusSeconds(60))
                .expiresAt(expires).claim("roles", List.of(role)).build();
        return encoder.encode(JwtEncoderParameters.from(JwsHeader.with(MacAlgorithm.HS256).build(), claims)).getTokenValue();
    }

    @Test
    void everyUnauthorizedRequestIsRejectedAndAllowedOnesPass() throws Exception {
        String admin = login("admin", "admin-pass");
        String service = login("jobboard", "jobboard-pass");
        String alice = login("alice", "alice-pass");
        String bob = login("bob", "bob-pass");
        // same header and claims as a real token, last signature character flipped
        String tampered = alice.substring(0, alice.length() - 2) + (alice.endsWith("A") ? "BB" : "AA");
        String expired = forged("pathrank-ads", Instant.now().minusSeconds(3600), "ADMIN");
        String wrongIssuer = forged("someone-else", Instant.now().plusSeconds(3600), "ADMIN");
        String auction = "{\"userId\": 1, \"organic\": [{\"jobId\": 1, \"score\": 0.1}]}";
        String click = "{\"impressionId\": \"nope\"}";

        List<Case> rejected = List.of(
                new Case("auction, no token", HttpMethod.POST, "/ads/auction", null, auction, 401),
                new Case("click, no token", HttpMethod.POST, "/ads/click", null, click, 401),
                new Case("campaigns, no token", HttpMethod.GET, "/api/campaigns", null, null, 401),
                new Case("stats, no token", HttpMethod.GET, "/ads/stats", null, null, 401),
                new Case("admin, no token", HttpMethod.GET, "/admin/budget/adv-1", null, null, 401),
                new Case("auction, garbage token", HttpMethod.POST, "/ads/auction", "not.a.jwt", auction, 401),
                new Case("auction, tampered signature", HttpMethod.POST, "/ads/auction", tampered, auction, 401),
                new Case("campaigns, expired token", HttpMethod.GET, "/api/campaigns", expired, null, 401),
                new Case("campaigns, wrong issuer", HttpMethod.GET, "/api/campaigns", wrongIssuer, null, 401),
                new Case("auction as advertiser", HttpMethod.POST, "/ads/auction", alice, auction, 403),
                new Case("click as advertiser", HttpMethod.POST, "/ads/click", alice, click, 403),
                new Case("stats as service", HttpMethod.GET, "/ads/stats", service, null, 403),
                new Case("stats as advertiser", HttpMethod.GET, "/ads/stats", alice, null, 403),
                new Case("admin budget as advertiser", HttpMethod.GET, "/admin/budget/adv-1", alice, null, 403),
                new Case("console API as service", HttpMethod.GET, "/api/campaigns", service, null, 403),
                new Case("alice reads bob's campaign", HttpMethod.GET, "/api/campaigns/ad-2", alice, null, 403),
                new Case("alice reads bob's campaign stats", HttpMethod.GET, "/api/campaigns/ad-2/stats", alice, null, 403),
                new Case("bob reads alice's spend", HttpMethod.GET, "/api/advertisers/adv-1/spend", bob, null, 403),
                new Case("unmapped path", HttpMethod.GET, "/internal/anything", admin, null, 403));
        List<Case> allowed = List.of(
                new Case("auction as service", HttpMethod.POST, "/ads/auction", service, auction, 200),
                new Case("alice reads her campaign", HttpMethod.GET, "/api/campaigns/ad-1", alice, null, 200),
                new Case("alice reads her spend", HttpMethod.GET, "/api/advertisers/adv-1/spend", alice, null, 200),
                new Case("admin reads any campaign", HttpMethod.GET, "/api/campaigns/ad-2", admin, null, 200),
                new Case("admin reads stats", HttpMethod.GET, "/ads/stats", admin, null, 200),
                new Case("health is public", HttpMethod.GET, "/actuator/health", null, null, 200),
                new Case("missing campaign is 404, not 403", HttpMethod.GET, "/api/campaigns/ad-999", admin, null, 404));

        List<String> failures = new ArrayList<>();
        for (Case c : concat(rejected, allowed)) {
            ResponseEntity<String> r = http.exchange(c.path(), c.method(), json(c.body(), c.token()), String.class);
            if (r.getStatusCode().value() != c.expected()) {
                failures.add(c.name() + ": expected " + c.expected() + ", got " + r.getStatusCode().value());
            }
        }
        assertTrue(failures.isEmpty(), String.join("\n", failures));
        System.out.printf("[security] rejected %d/%d unauthorized requests, allowed %d/%d legitimate ones%n",
                rejected.size(), rejected.size(), allowed.size(), allowed.size());
    }

    @Test
    void advertiserListOnlyContainsOwnCampaigns() throws Exception {
        String alice = login("alice", "alice-pass");
        ResponseEntity<String> r = http.exchange("/api/campaigns", HttpMethod.GET, json(null, alice), String.class);
        assertEquals(200, r.getStatusCode().value());
        JsonNode list = mapper.readTree(r.getBody());
        assertTrue(list.size() > 0);
        for (JsonNode c : list) assertEquals("adv-1", c.get("advertiserId").asText());
    }

    @Test
    void wrongPasswordIs401AndSaysNothingElse() throws Exception {
        ResponseEntity<String> r = http.postForEntity("/auth/token",
                json(Map.of("username", "alice", "password", "wrong"), null), String.class);
        assertEquals(401, r.getStatusCode().value());
        ResponseEntity<String> u = http.postForEntity("/auth/token",
                json(Map.of("username", "nobody", "password", "x"), null), String.class);
        assertEquals(401, u.getStatusCode().value());
        assertNotNull(r);
    }

    private static <T> List<T> concat(List<T> a, List<T> b) {
        List<T> out = new ArrayList<>(a);
        out.addAll(b);
        return out;
    }
}
