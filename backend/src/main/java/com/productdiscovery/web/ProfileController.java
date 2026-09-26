package com.productdiscovery.web;

import com.productdiscovery.service.ProfileService;
import com.productdiscovery.service.ProfileSnapshot;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/users/{userId}/profile")
public class ProfileController {
    private final ProfileService profiles;
    public ProfileController(ProfileService profiles) { this.profiles = profiles; }
    @GetMapping public ProfileSnapshot get(@PathVariable Long userId) { return profiles.get(userId); }
    @PostMapping("/rebuild") public ProfileSnapshot rebuild(@PathVariable Long userId) { return profiles.rebuild(userId); }
}
