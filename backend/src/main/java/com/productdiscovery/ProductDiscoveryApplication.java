package com.productdiscovery;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class ProductDiscoveryApplication {
    public static void main(String[] args) {
        SpringApplication.run(ProductDiscoveryApplication.class, args);
    }
}
