package com.swiftvalidator.entity;

import jakarta.persistence.*;
import lombok.*;

import java.math.BigDecimal;

@Entity
@Table(name = "ds_risk_scores")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DsRiskScore {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "analysis_id", nullable = false)
    private SwiftAnalysis analysis;

    @Column(name = "risk_level", length = 20)
    private String riskLevel;

    @Column(name = "risk_score", precision = 6, scale = 4)
    private BigDecimal riskScore;

    @Column(name = "ml_risk_score", precision = 6, scale = 4)
    private BigDecimal mlRiskScore;

    @Column(name = "model_used", length = 30)
    private String modelUsed;

    @Column(name = "completeness_score")
    private Integer completenessScore;

    @Column(name = "n_clauses")
    private Integer nClauses;

    @Column(name = "n_compliance_clauses")
    private Integer nComplianceClauses;

    @Column(name = "n_critical_clauses")
    private Integer nCriticalClauses;

    @Column(name = "n_high_risk_clauses")
    private Integer nHighRiskClauses;

    @Column(name = "n_documents_required")
    private Integer nDocumentsRequired;

    @Column(name = "incoterm_risk")
    private Integer incotermRisk;

    @Column(name = "presentation_days")
    private Integer presentationDays;

    @Column(name = "validity_days")
    private Integer validityDays;

    @Column(name = "processing_ms")
    private Long processingMs;
}
