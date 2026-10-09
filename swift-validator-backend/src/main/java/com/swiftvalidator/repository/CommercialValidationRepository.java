package com.swiftvalidator.repository;

import com.swiftvalidator.entity.CommercialValidation;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface CommercialValidationRepository extends JpaRepository<CommercialValidation, Long> {

    Page<CommercialValidation> findByUserIdOrderByCreatedAtDesc(Long userId, Pageable pageable);

    /** Historique aligné sur un MT700 : validations commerciales rattachées à l'analyse SWIFT. */
    Page<CommercialValidation> findBySwiftAnalysisIdAndUserIdOrderByCreatedAtDesc(
            Long swiftAnalysisId, Long userId, Pageable pageable);

    Optional<CommercialValidation> findByIdAndUserId(Long id, Long userId);

    long countByUserId(Long userId);

    @Query("""
        SELECT v FROM CommercialValidation v
        WHERE v.user.id = :userId
        AND (:riskLevel IS NULL OR v.riskLevel = :riskLevel)
        AND (:lcReference IS NULL OR v.lcReference LIKE %:lcReference%)
        ORDER BY v.createdAt DESC
    """)
    Page<CommercialValidation> search(
        @Param("userId") Long userId,
        @Param("riskLevel") String riskLevel,
        @Param("lcReference") String lcReference,
        Pageable pageable
    );
}
