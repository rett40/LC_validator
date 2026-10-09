package com.swiftvalidator.repository;

import com.swiftvalidator.entity.AnalysisReport;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface AnalysisReportRepository extends JpaRepository<AnalysisReport, Long> {

    List<AnalysisReport> findByAnalysisIdOrderByGeneratedAtDesc(Long analysisId);

    boolean existsByAnalysisIdAndReportType(Long analysisId, String reportType);
}
