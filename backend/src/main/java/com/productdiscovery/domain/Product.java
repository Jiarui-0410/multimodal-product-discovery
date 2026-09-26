package com.productdiscovery.domain;

import jakarta.persistence.*;

@Entity
@Table(name = "products")
public class Product {
    @Id
    @Column(name = "product_id")
    private Long id;

    @Column(name = "ml_index_id", nullable = false, unique = true)
    private Long mlIndexId;

    @Column(nullable = false, length = 500)
    private String name;
    @Column(length = 150)
    private String category;
    @Column(length = 100)
    private String color;
    @Column(length = 100)
    private String season;
    @Column(length = 100)
    private String usage;
    @Column(length = 100)
    private String gender;

    @Column(name = "image_path", length = 1000)
    private String imagePath;

    @Column(name = "semantic_embedding", columnDefinition = "TEXT")
    private String semanticEmbedding;

    protected Product() {}

    public Product(Long id, Long mlIndexId, String name, String category, String color,
                   String season, String usage, String gender, String imagePath) {
        this.id = id;
        this.mlIndexId = mlIndexId;
        this.name = name;
        this.category = category;
        this.color = color;
        this.season = season;
        this.usage = usage;
        this.gender = gender;
        this.imagePath = imagePath;
    }

    public Long getId() { return id; }
    public Long getMlIndexId() { return mlIndexId; }
    public String getName() { return name; }
    public String getCategory() { return category; }
    public String getColor() { return color; }
    public String getSeason() { return season; }
    public String getUsage() { return usage; }
    public String getGender() { return gender; }
    public String getImagePath() { return imagePath; }
    public String getSemanticEmbedding() { return semanticEmbedding; }
    public void setSemanticEmbedding(String semanticEmbedding) { this.semanticEmbedding = semanticEmbedding; }
}
