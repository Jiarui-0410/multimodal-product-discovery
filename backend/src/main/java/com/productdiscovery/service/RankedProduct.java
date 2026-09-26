package com.productdiscovery.service;

public record RankedProduct(
        Long productId,
        String name,
        String category,
        String color,
        String season,
        String usage,
        String gender,
        String imagePath,
        double queryScore,
        double userPreferenceScore,
        double metadataPreferenceScore,
        double finalScore
) {}
