package com.pathrank.ads.server.security;

import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * ads.security.*: whether auth is on, the HS256 signing secret, token lifetime, and the demo users.
 * The users in application.yml are for local runs only; a real deployment would put them behind an
 * identity provider and only keep the resource-server half of this.
 */
@ConfigurationProperties("ads.security")
public record SecurityProperties(
        @DefaultValue("true") boolean enabled,
        String jwtSecret,
        @DefaultValue("3600") long tokenTtlSeconds,
        @DefaultValue("pathrank-ads") String issuer,
        List<User> users) {

    /** role is one of ADMIN, SERVICE, ADVERTISER; advertiserId is required for ADVERTISER. */
    public record User(String username, String password, String role, String advertiserId) {}

    public List<User> usersOrEmpty() {
        return users == null ? List.of() : users;
    }
}
