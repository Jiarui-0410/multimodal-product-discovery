package com.productdiscovery.service;

import com.productdiscovery.domain.InteractionType;
import java.time.Instant;
import java.util.List;
import java.util.Map;

public record ProfileSnapshot(
        Long userId,
        Map<String, Double> topCategories,
        Map<String, Double> colorPreferences,
        Map<String, Double> usagePreferences,
        Map<InteractionType, Long> interactionStatistics,
        List<Double> semanticEmbedding,
        Instant updatedAt
) {
    public static ProfileSnapshot empty(Long userId) {
        return new ProfileSnapshot(userId, Map.of(), Map.of(), Map.of(), Map.of(), List.of(), Instant.now());
    }
}
