package com.productdiscovery.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "user_profiles")
public class UserProfile {
    @Id
    @Column(name = "user_id")
    private Long userId;

    @OneToOne(fetch = FetchType.LAZY, optional = false)
    @MapsId
    @JoinColumn(name = "user_id")
    private UserAccount user;

    @Column(name = "category_preferences", nullable = false, columnDefinition = "TEXT")
    private String categoryPreferences = "{}";
    @Column(name = "color_preferences", nullable = false, columnDefinition = "TEXT")
    private String colorPreferences = "{}";
    @Column(name = "usage_preferences", nullable = false, columnDefinition = "TEXT")
    private String usagePreferences = "{}";
    @Column(name = "semantic_embedding", columnDefinition = "TEXT")
    private String semanticEmbedding;
    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt = Instant.now();

    protected UserProfile() {}
    public UserProfile(UserAccount user) { this.user = user; }

    public void update(String categories, String colors, String usages, String embedding) {
        this.categoryPreferences = categories;
        this.colorPreferences = colors;
        this.usagePreferences = usages;
        this.semanticEmbedding = embedding;
        this.updatedAt = Instant.now();
    }
    public Long getUserId() { return userId; }
    public String getCategoryPreferences() { return categoryPreferences; }
    public String getColorPreferences() { return colorPreferences; }
    public String getUsagePreferences() { return usagePreferences; }
    public String getSemanticEmbedding() { return semanticEmbedding; }
    public Instant getUpdatedAt() { return updatedAt; }
}
