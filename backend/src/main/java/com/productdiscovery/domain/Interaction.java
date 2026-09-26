package com.productdiscovery.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "interactions")
public class Interaction {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "interaction_id")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id")
    private UserAccount user;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "product_id")
    private Product product;

    @Enumerated(EnumType.STRING)
    @Column(name = "interaction_type", nullable = false, length = 20)
    private InteractionType type;

    @Column(name = "occurred_at", nullable = false)
    private Instant occurredAt = Instant.now();

    protected Interaction() {}
    public Interaction(UserAccount user, Product product, InteractionType type) {
        this.user = user;
        this.product = product;
        this.type = type;
    }
    public Long getId() { return id; }
    public Product getProduct() { return product; }
    public InteractionType getType() { return type; }
    public Instant getOccurredAt() { return occurredAt; }
}
