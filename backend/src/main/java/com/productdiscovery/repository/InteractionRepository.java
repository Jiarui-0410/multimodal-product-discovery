package com.productdiscovery.repository;

import com.productdiscovery.domain.Interaction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import java.util.List;

public interface InteractionRepository extends JpaRepository<Interaction, Long> {
    @Query("select i from Interaction i join fetch i.product where i.user.id = :userId order by i.occurredAt desc")
    List<Interaction> findForProfile(@Param("userId") Long userId);
}
