package com.productdiscovery.service;

import java.util.List;

public record SearchResponse(String mode, List<RankedProduct> results) {}
