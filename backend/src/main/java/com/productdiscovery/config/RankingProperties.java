package com.productdiscovery.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "ranking")
public record RankingProperties(
        double queryWeight,
        double userWeight,
        double metadataWeight,
        int candidateMultiplier
) {
    public RankingProperties {
        if (!Double.isFinite(queryWeight) || !Double.isFinite(userWeight)
                || !Double.isFinite(metadataWeight)
                || queryWeight < 0 || userWeight < 0 || metadataWeight < 0) {
            throw new IllegalArgumentException("Ranking weights must be finite and non-negative");
        }
        double sum = queryWeight + userWeight + metadataWeight;
        if (Math.abs(sum - 1.0) > 0.0001) {
            throw new IllegalArgumentException("Ranking weights must sum to 1.0");
        }
        if (candidateMultiplier < 1) {
            throw new IllegalArgumentException("Candidate multiplier must be at least 1");
        }
    }
}
