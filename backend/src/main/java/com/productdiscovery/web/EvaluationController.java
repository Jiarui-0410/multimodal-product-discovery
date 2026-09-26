package com.productdiscovery.web;

import com.productdiscovery.service.MetricsService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Set;

@RestController
@RequestMapping("/api/evaluation")
public class EvaluationController {
    private final MetricsService metrics;
    public EvaluationController(MetricsService metrics) { this.metrics = metrics; }

    @PostMapping("/metrics")
    public MetricsService.RankingMetrics calculate(@Valid @RequestBody MetricsRequest request) {
        return metrics.calculate(request.rankedProductIds(), request.relevantProductIds(), request.k());
    }

    public record MetricsRequest(
            @NotEmpty List<Long> rankedProductIds,
            @NotEmpty Set<Long> relevantProductIds,
            @Min(1) int k
    ) {}
}
