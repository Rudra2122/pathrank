package com.pathrank.ads.server.security;

/** The three kinds of caller. */
public final class Roles {
    private Roles() {}

    /** Sees and manages everything, including /ads/stats and /admin/**. */
    public static final String ADMIN = "ADMIN";
    /** The job board backend: may call /ads/auction and /ads/click, nothing else. */
    public static final String SERVICE = "SERVICE";
    /** An advertiser: may read only its own campaigns and spend through /api/**. */
    public static final String ADVERTISER = "ADVERTISER";

    public static final String CLAIM_ROLES = "roles";
    public static final String CLAIM_ADVERTISER = "advertiser_id";
}
