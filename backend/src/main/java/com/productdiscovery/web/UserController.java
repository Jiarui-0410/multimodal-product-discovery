package com.productdiscovery.web;

import com.productdiscovery.domain.UserAccount;
import com.productdiscovery.repository.UserRepository;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/users")
public class UserController {
    private final UserRepository users;
    public UserController(UserRepository users) { this.users = users; }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public UserView create(@Valid @RequestBody CreateUserRequest request) {
        String username = request.username().trim();
        if (users.existsByUsernameIgnoreCase(username)) throw new IllegalArgumentException("Username already exists");
        return UserView.from(users.save(new UserAccount(username)));
    }

    @GetMapping
    public List<UserView> list() { return users.findAll().stream().map(UserView::from).toList(); }

    public record CreateUserRequest(@NotBlank @Size(max = 100) String username) {}
    public record UserView(Long userId, String username) {
        static UserView from(UserAccount user) { return new UserView(user.getId(), user.getUsername()); }
    }
}
