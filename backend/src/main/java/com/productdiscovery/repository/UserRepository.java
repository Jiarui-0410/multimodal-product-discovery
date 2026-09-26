package com.productdiscovery.repository;

import com.productdiscovery.domain.UserAccount;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<UserAccount, Long> {
    boolean existsByUsernameIgnoreCase(String username);
}
