package com.swiftvalidator.repository;

import com.swiftvalidator.entity.UcpFinding;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface UcpFindingRepository extends JpaRepository<UcpFinding, Long> {
    List<UcpFinding> findByAnalysisIdOrderBySeverityAscTierAsc(Long analysisId);
    List<UcpFinding> findByAnalysisIdAndBlocksPaymentTrue(Long analysisId);
}
