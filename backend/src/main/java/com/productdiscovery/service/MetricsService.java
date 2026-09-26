package com.productdiscovery.service;

import org.springframework.stereotype.Service;
import java.util.*;

@Service
public class MetricsService {
    public RankingMetrics calculate(List<Long> rankedProductIds, Set<Long> relevantProductIds, int k) {
        int limit = Math.min(k, rankedProductIds.size());
        int hits = 0;
        double dcg = 0.0;
        for (int i = 0; i < limit; i++) {
            if (relevantProductIds.contains(rankedProductIds.get(i))) {
                hits++;
                dcg += 1.0 / log2(i + 2.0);
            }
        }
        int idealHits = Math.min(k, relevantProductIds.size());
        double idcg = 0.0;
        for (int i = 0; i < idealHits; i++) idcg += 1.0 / log2(i + 2.0);
        double precision = k == 0 ? 0.0 : (double) hits / k;
        double recall = relevantProductIds.isEmpty() ? 0.0 : (double) hits / relevantProductIds.size();
        double ndcg = idcg == 0.0 ? 0.0 : dcg / idcg;
        return new RankingMetrics(k, precision, recall, ndcg, hits);
    }

    private static double log2(double value) { return Math.log(value) / Math.log(2.0); }
    public record RankingMetrics(int k, double precisionAtK, double recallAtK, double ndcgAtK, int hits) {}
}
