package com.productdiscovery.domain;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "search_history")
public class SearchHistory {
    public enum QueryType { TEXT, IMAGE }

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "search_id")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id")
    private UserAccount user;

    @Column(name = "query_text", length = 1000)
    private String queryText;

    @Enumerated(EnumType.STRING)
    @Column(name = "query_type", nullable = false, length = 20)
    private QueryType queryType;

    @Column(name = "searched_at", nullable = false)
    private Instant searchedAt = Instant.now();

    protected SearchHistory() {}
    public SearchHistory(UserAccount user, String queryText, QueryType queryType) {
        this.user = user;
        this.queryText = queryText;
        this.queryType = queryType;
    }
}
