package com.pathrank.ads.core;

/** Where the auction gets a predicted click-through rate for an ad. */
@FunctionalInterface
public interface CtrSource {
    double pctr(String adId);
}
