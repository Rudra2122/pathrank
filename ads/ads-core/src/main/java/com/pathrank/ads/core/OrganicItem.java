package com.pathrank.ads.core;

/** One organic (non-paid) result from the PathRank ranker: a job and its score (treated as an apply probability). */
public record OrganicItem(long jobId, double score) {}
