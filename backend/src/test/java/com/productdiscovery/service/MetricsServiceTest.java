package com.productdiscovery.service;

import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Set;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class MetricsServiceTest {
    private final MetricsService metrics = new MetricsService();

    @Test
    void calculatesBinaryRankingMetricsAtK() {
        var result = metrics.calculate(List.of(1L, 2L, 3L, 4L), Set.of(2L, 4L), 3);
        assertThat(result.hits()).isEqualTo(1);
        assertThat(result.precisionAtK()).isCloseTo(1.0 / 3.0, within(0.00001));
        assertThat(result.recallAtK()).isEqualTo(0.5);
        assertThat(result.ndcgAtK()).isBetween(0.0, 1.0);
    }
}
