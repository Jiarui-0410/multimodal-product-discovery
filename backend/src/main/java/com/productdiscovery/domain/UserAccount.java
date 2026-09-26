package com.productdiscovery.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "users")
public class UserAccount {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_id")
    private Long id;

    @Column(nullable = false, unique = true, length = 100)
    private String username;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt = Instant.now();

    protected UserAccount() {}

    public UserAccount(String username) { this.username = username; }
    public Long getId() { return id; }
    public String getUsername() { return username; }
    public Instant getCreatedAt() { return createdAt; }
}
