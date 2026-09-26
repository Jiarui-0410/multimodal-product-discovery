package com.productdiscovery.service;

import com.productdiscovery.config.RankingProperties;
import com.productdiscovery.domain.Product;
import com.productdiscovery.domain.SearchHistory;
import com.productdiscovery.domain.UserAccount;
import com.productdiscovery.ml.MlClient;
import com.productdiscovery.repository.ProductRepository;
import com.productdiscovery.repository.SearchHistoryRepository;
import com.productdiscovery.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

@Service
public class SearchService {
    private final MlClient ml;
    private final UserRepository users;
    private final ProductRepository products;
    private final SearchHistoryRepository history;
    private final ProfileService profiles;
    private final JsonVectors json;
    private final RankingProperties weights;

    public SearchService(MlClient ml, UserRepository users, ProductRepository products,
                         SearchHistoryRepository history, ProfileService profiles,
                         JsonVectors json, RankingProperties weights) {
        this.ml = ml;
        this.users = users;
        this.products = products;
        this.history = history;
        this.profiles = profiles;
        this.json = json;
        this.weights = weights;
    }

    @Transactional
    public SearchResponse searchText(Long userId, String query, int topK, boolean personalized) {
        UserAccount user = requireUser(userId);
        history.save(new SearchHistory(user, query, SearchHistory.QueryType.TEXT));
        return rank(userId, topK, personalized,
                () -> ml.retrieveText(query, candidateLimit(topK)));
    }

    @Transactional
    public SearchResponse searchImage(Long userId, byte[] bytes, String contentType,
                                      String filename, int topK, boolean personalized) {
        UserAccount user = requireUser(userId);
        history.save(new SearchHistory(user, null, SearchHistory.QueryType.IMAGE));
        return rank(userId, topK, personalized,
                () -> ml.retrieveImage(bytes, contentType, filename, candidateLimit(topK)));
    }

    private SearchResponse rank(Long userId, int topK, boolean personalized,
                                Supplier<List<MlClient.RetrievalHit>> retrieval) {
        List<MlClient.RetrievalHit> hits = retrieval.get();
        Set<Long> indexIds = hits.stream().map(MlClient.RetrievalHit::indexId).collect(Collectors.toSet());
        Map<Long, Product> catalog = products.findByMlIndexIdIn(indexIds).stream()
                .collect(Collectors.toMap(Product::getMlIndexId, p -> p));
        ProfileSnapshot profile = personalized ? profiles.get(userId) : ProfileSnapshot.empty(userId);
        List<Product> embeddingsToPersist = new ArrayList<>();
        List<RankedProduct> ranked = new ArrayList<>();

        for (MlClient.RetrievalHit hit : hits) {
            Product product = catalog.get(hit.indexId());
            if (product == null) continue;
            if (hit.embedding() != null && !hit.embedding().isEmpty()) {
                String canonicalEmbedding = json.write(hit.embedding());
                if (!canonicalEmbedding.equals(product.getSemanticEmbedding())) {
                    product.setSemanticEmbedding(canonicalEmbedding);
                    embeddingsToPersist.add(product);
                }
            }
            double queryScore = cosineToUnitInterval(hit.score());
            double userScore = cosineToUnitInterval(cosine(profile.semanticEmbedding(), hit.embedding()));
            double metadataScore = metadataScore(profile, product);
            double finalScore = personalized
                    ? weights.queryWeight() * queryScore
                      + weights.userWeight() * userScore
                      + weights.metadataWeight() * metadataScore
                    : queryScore;
            ranked.add(new RankedProduct(product.getId(), product.getName(), product.getCategory(),
                    product.getColor(), product.getSeason(), product.getUsage(), product.getGender(),
                    product.getImagePath(), queryScore, userScore, metadataScore, finalScore));
        }
        if (!embeddingsToPersist.isEmpty()) products.saveAll(embeddingsToPersist);
        ranked.sort(Comparator.comparingDouble(RankedProduct::finalScore).reversed());
        return new SearchResponse(personalized ? "PERSONALIZED" : "CLIP_BASELINE",
                ranked.stream().limit(topK).toList());
    }

    private UserAccount requireUser(Long userId) {
        return users.findById(userId)
                .orElseThrow(() -> new NoSuchElementException("User not found: " + userId));
    }
    private int candidateLimit(int topK) {
        return Math.min(500, Math.max(topK, topK * weights.candidateMultiplier()));
    }
    private static double cosineToUnitInterval(double score) {
        return Math.max(0.0, Math.min(1.0, (score + 1.0) / 2.0));
    }
    private static double cosine(List<Double> left, List<Double> right) {
        if (left == null || right == null || left.isEmpty() || left.size() != right.size()) return -1.0;
        double dot = 0, leftNorm = 0, rightNorm = 0;
        for (int i = 0; i < left.size(); i++) {
            dot += left.get(i) * right.get(i);
            leftNorm += left.get(i) * left.get(i);
            rightNorm += right.get(i) * right.get(i);
        }
        return leftNorm == 0 || rightNorm == 0 ? -1.0 : dot / Math.sqrt(leftNorm * rightNorm);
    }
    private static double metadataScore(ProfileSnapshot profile, Product product) {
        return 0.5 * profile.topCategories().getOrDefault(product.getCategory(), 0.0)
                + 0.3 * profile.colorPreferences().getOrDefault(product.getColor(), 0.0)
                + 0.2 * profile.usagePreferences().getOrDefault(product.getUsage(), 0.0);
    }
}
