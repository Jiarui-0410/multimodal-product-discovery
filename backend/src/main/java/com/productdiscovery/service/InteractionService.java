package com.productdiscovery.service;

import com.productdiscovery.domain.*;
import com.productdiscovery.repository.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.NoSuchElementException;

@Service
public class InteractionService {
    private final UserRepository users;
    private final ProductRepository products;
    private final InteractionRepository interactions;
    private final ProfileService profiles;

    public InteractionService(UserRepository users, ProductRepository products,
                              InteractionRepository interactions, ProfileService profiles) {
        this.users = users;
        this.products = products;
        this.interactions = interactions;
        this.profiles = profiles;
    }

    @Transactional
    public ProfileSnapshot record(Long userId, Long productId, InteractionType type) {
        UserAccount user = users.findById(userId)
                .orElseThrow(() -> new NoSuchElementException("User not found: " + userId));
        Product product = products.findById(productId)
                .orElseThrow(() -> new NoSuchElementException("Product not found: " + productId));
        interactions.save(new Interaction(user, product, type));
        return profiles.rebuild(userId);
    }
}
