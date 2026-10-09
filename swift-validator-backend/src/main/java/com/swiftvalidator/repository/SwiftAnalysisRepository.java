package com.swiftvalidator.repository;

import com.swiftvalidator.entity.SwiftAnalysis;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface SwiftAnalysisRepository extends JpaRepository<SwiftAnalysis, Long> {

    Page<SwiftAnalysis> findByUserIdOrderByCreatedAtDesc(Long userId, Pageable pageable);

    Optional<SwiftAnalysis> findByLcNumber(String lcNumber);
    Optional<SwiftAnalysis> findByIdAndUserId(Long id, Long userId);

    Page<SwiftAnalysis> findByRiskLevelOrderByCreatedAtDesc(String riskLevel, Pageable pageable);

    @Query("SELECT a FROM SwiftAnalysis a WHERE a.user.id = :userId " +
           "AND (:riskLevel IS NULL OR a.riskLevel = :riskLevel) " +
           "AND (:lcNumber IS NULL OR LOWER(a.lcNumber) LIKE LOWER(CONCAT('%', :lcNumber, '%'))) " +
           "ORDER BY a.createdAt DESC")
    Page<SwiftAnalysis> search(
        @Param("userId") Long userId,
        @Param("riskLevel") String riskLevel,
        @Param("lcNumber") String lcNumber,
        Pageable pageable
    );

    @Query("SELECT COUNT(a) FROM SwiftAnalysis a WHERE a.user.id = :userId AND a.riskLevel = :level")
    long countByUserIdAndRiskLevel(@Param("userId") Long userId, @Param("level") String level);
    long countByUserId(Long userId);

    @Query("SELECT COUNT(a) FROM SwiftAnalysis a WHERE a.createdAt >= :since")
    long countSince(@Param("since") LocalDateTime since);

    @Query("SELECT AVG(a.ucpComplianceScore) FROM SwiftAnalysis a WHERE a.user.id = :userId")
    Double avgUcpScore(@Param("userId") Long userId);

    // ── Admin queries ──

    @Query("SELECT COUNT(a) FROM SwiftAnalysis a WHERE a.riskLevel IN :levels")
    long countByRiskLevelIn(@Param("levels") List<String> levels);

    @Query("SELECT AVG(a.ucpComplianceScore) FROM SwiftAnalysis a")
    Double avgUcpScoreAll();

    @Query("SELECT a FROM SwiftAnalysis a WHERE a.user.id = :userId ORDER BY a.createdAt DESC")
    List<SwiftAnalysis> findByUserIdOrdered(@Param("userId") Long userId, org.springframework.data.domain.Pageable pageable);

    @Query(value = "SELECT DATE(created_at) as day, COUNT(*) as cnt " +
                   "FROM swift_analyses WHERE created_at >= :since " +
                   "GROUP BY DATE(created_at) ORDER BY day",
           nativeQuery = true)
    List<Object[]> countByDaySince(@Param("since") LocalDateTime since);
}
