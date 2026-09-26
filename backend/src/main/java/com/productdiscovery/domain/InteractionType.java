package com.productdiscovery.domain;

public enum InteractionType {
    VIEW(1.0), CLICK(2.0), LIKE(4.0), SAVE(5.0), PURCHASE(8.0);

    private final double weight;
    InteractionType(double weight) { this.weight = weight; }
    public double weight() { return weight; }
}
