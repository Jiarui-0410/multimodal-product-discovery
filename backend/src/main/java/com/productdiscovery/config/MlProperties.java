package com.productdiscovery.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "ml")
public record MlProperties(String baseUrl) {}
