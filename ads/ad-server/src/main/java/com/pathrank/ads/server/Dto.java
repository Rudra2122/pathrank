package com.pathrank.ads.server;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.util.List;

/** Request/response bodies for the REST API. */
public final class Dto {
    private Dto() {}

    public record Organic(long jobId, double score) {}

    public record AuctionRequest(@NotNull Long userId, @NotEmpty @Size(max = 200) List<@Valid Organic> organic) {}

    /** One row of the final list. Sponsored rows carry the ad, the impression id to click on, and the price. */
    public record Item(int rank, long jobId, double score, boolean sponsored, String adId, String impressionId,
                       Long priceMicros) {}

    public record AuctionResponse(String requestId, List<Item> items, int sponsoredCount, int candidates) {}

    public record ClickRequest(@NotBlank String impressionId) {}

    public record ClickResponse(String status, boolean charged, long priceMicros) {}
}
