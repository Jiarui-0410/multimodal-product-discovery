package com.productdiscovery.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RankingPropertiesTest {
    @Test
    void acceptsValidWeights() {
        assertDoesNotThrow(() -> new RankingProperties(0.6, 0.3, 0.1, 10));
    }

    @Test
    void rejectsNegativeWeightsEvenWhenTheySumToOne() {
        assertThrows(IllegalArgumentException.class,
                () -> new RankingProperties(1.1, -0.1, 0.0, 10));
    }
}
