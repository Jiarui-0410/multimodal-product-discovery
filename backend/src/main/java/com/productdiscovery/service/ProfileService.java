package com.productdiscovery.service;

import com.productdiscovery.domain.*;
import com.productdiscovery.repository.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.Instant;
import java.util.*;

@Service
public class ProfileService {
    private final UserRepository users;
    private final InteractionRepository interactions;
    private final UserProfileRepository profiles;
    private final JsonVectors json;

    public ProfileService(UserRepository users, InteractionRepository interactions,
                          UserProfileRepository profiles, JsonVectors json) {
        this.users = users;
        this.interactions = interactions;
        this.profiles = profiles;
        this.json = json;
    }

    @Transactional
    public ProfileSnapshot rebuild(Long userId) {
        UserAccount user = users.findById(userId)
                .orElseThrow(() -> new NoSuchElementException("User not found: " + userId));
        List<Interaction> events = interactions.findForProfile(userId);
        Map<String, Double> categories = new HashMap<>();
        Map<String, Double> colors = new HashMap<>();
        Map<String, Double> usages = new HashMap<>();
        EnumMap<InteractionType, Long> statistics = new EnumMap<>(InteractionType.class);
        double[] vector = null;
        double vectorWeight = 0.0;

        for (Interaction event : events) {
            double weight = event.getType().weight();
            Product product = event.getProduct();
            add(categories, product.getCategory(), weight);
            add(colors, product.getColor(), weight);
            add(usages, product.getUsage(), weight);
            statistics.merge(event.getType(), 1L, Long::sum);
            List<Double> embedding = json.readVector(product.getSemanticEmbedding());
            if (!embedding.isEmpty()) {
                if (vector == null) vector = new double[embedding.size()];
                if (vector.length == embedding.size()) {
                    for (int i = 0; i < vector.length; i++) vector[i] += weight * embedding.get(i);
                    vectorWeight += weight;
                }
            }
        }

        List<Double> semantic = vector == null ? List.of() : normalize(vector, vectorWeight);
        Map<String, Double> normalizedCategories = normalizeMap(categories);
        Map<String, Double> normalizedColors = normalizeMap(colors);
        Map<String, Double> normalizedUsages = normalizeMap(usages);
        UserProfile profile = profiles.findById(userId).orElseGet(() -> new UserProfile(user));
        profile.update(json.write(normalizedCategories), json.write(normalizedColors),
                json.write(normalizedUsages), semantic.isEmpty() ? null : json.write(semantic));
        profiles.save(profile);
        return new ProfileSnapshot(userId, normalizedCategories, normalizedColors, normalizedUsages,
                statistics, semantic, profile.getUpdatedAt());
    }

    @Transactional(readOnly = true)
    public ProfileSnapshot get(Long userId) {
        if (!users.existsById(userId)) throw new NoSuchElementException("User not found: " + userId);
        UserProfile profile = profiles.findById(userId).orElse(null);
        if (profile == null) return ProfileSnapshot.empty(userId);
        EnumMap<InteractionType, Long> stats = new EnumMap<>(InteractionType.class);
        for (Interaction event : interactions.findForProfile(userId)) stats.merge(event.getType(), 1L, Long::sum);
        return new ProfileSnapshot(userId, json.readMap(profile.getCategoryPreferences()),
                json.readMap(profile.getColorPreferences()), json.readMap(profile.getUsagePreferences()),
                stats, json.readVector(profile.getSemanticEmbedding()), profile.getUpdatedAt());
    }

    private static void add(Map<String, Double> map, String key, double weight) {
        if (key != null && !key.isBlank()) map.merge(key, weight, Double::sum);
    }
    private static Map<String, Double> normalizeMap(Map<String, Double> values) {
        double total = values.values().stream().mapToDouble(Double::doubleValue).sum();
        if (total == 0) return Map.of();
        LinkedHashMap<String, Double> result = new LinkedHashMap<>();
        values.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .forEach(entry -> result.put(entry.getKey(), entry.getValue() / total));
        return result;
    }
    private static List<Double> normalize(double[] vector, double weight) {
        if (weight == 0) return List.of();
        double norm = 0;
        for (int i = 0; i < vector.length; i++) { vector[i] /= weight; norm += vector[i] * vector[i]; }
        norm = Math.sqrt(norm);
        if (norm == 0) return List.of();
        List<Double> result = new ArrayList<>(vector.length);
        for (double value : vector) result.add(value / norm);
        return result;
    }
}
