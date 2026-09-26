package com.productdiscovery.web;

import com.productdiscovery.domain.InteractionType;
import com.productdiscovery.service.InteractionService;
import com.productdiscovery.service.ProfileSnapshot;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/interactions")
public class InteractionController {
    private final InteractionService interactions;
    public InteractionController(InteractionService interactions) { this.interactions = interactions; }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public ProfileSnapshot record(@Valid @RequestBody InteractionRequest request) {
        return interactions.record(request.userId(), request.productId(), request.type());
    }
    public record InteractionRequest(@NotNull Long userId, @NotNull Long productId,
                                     @NotNull InteractionType type) {}
}
