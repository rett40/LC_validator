package com.swiftvalidator.repository;

import com.swiftvalidator.entity.NlpClause;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface NlpClauseRepository extends JpaRepository<NlpClause, Long> {
    List<NlpClause> findByAnalysisIdOrderByImportanceAsc(Long analysisId);
    List<NlpClause> findByAnalysisIdAndRiskLevel(Long analysisId, String riskLevel);
}
