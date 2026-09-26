package com.productdiscovery.repository;

import com.productdiscovery.domain.Product;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Collection;
import java.util.List;

public interface ProductRepository extends JpaRepository<Product, Long> {
    List<Product> findByMlIndexIdIn(Collection<Long> mlIndexIds);
}
