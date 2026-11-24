package com.pathrank.ads.server.security;

import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;

/** Who is calling, read from the verified JWT. With security off, every caller is treated as admin. */
public record Caller(String username, boolean admin, String advertiserId) {

    public static Caller from(Authentication auth) {
        if (!(auth instanceof JwtAuthenticationToken jwt)) return new Caller("anonymous", true, null);
        boolean admin = auth.getAuthorities().stream().anyMatch(a -> a.getAuthority().equals("ROLE_" + Roles.ADMIN));
        return new Caller(jwt.getName(), admin, jwt.getToken().getClaimAsString(Roles.CLAIM_ADVERTISER));
    }

    /** Admins see everything; an advertiser only its own rows. */
    public boolean canSee(String ownerAdvertiserId) {
        return admin || (advertiserId != null && advertiserId.equals(ownerAdvertiserId));
    }
}
