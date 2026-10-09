package com.swiftvalidator.entity;

import jakarta.persistence.*;
import lombok.*;

import java.math.BigDecimal;

@Entity
@Table(name = "ucp_findings")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UcpFinding {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "analysis_id", nullable = false)
    private SwiftAnalysis analysis;

    @Column(name = "article", length = 20)
    private String article;

    @Column(name = "article_title", length = 200)
    private String articleTitle;

    @Column(name = "code", length = 60)
    private String code;

    @Column(name = "category", length = 30)
    private String category;

    @Column(name = "severity", length = 20)
    private String severity;

    @Column(name = "tier")
    private Integer tier;

    @Column(name = "weight", precision = 4, scale = 2)
    private BigDecimal weight;

    @Column(name = "message", columnDefinition = "TEXT")
    private String message;

    @Column(name = "clause_text", columnDefinition = "TEXT")
    private String clauseText;

    @Column(name = "recommendation", columnDefinition = "TEXT")
    private String recommendation;

    @Column(name = "blocks_payment")
    private Boolean blocksPayment;

    @Column(name = "cascade_impact")
    private Integer cascadeImpact;

    @Column(name = "exclusion_type", length = 30)
    private String exclusionType;

    @Column(name = "confidence", precision = 4, scale = 3)
    private BigDecimal confidence;

    /** true_violation | contractual_modification | operational_discrepancy */
    @Column(name = "finding_nature", length = 40)
    private String findingNature;

    /** HIGH | MEDIUM | LOW */
    @Column(name = "impact_level", length = 10)
    private String impactLevel;

    /** financial | transport | authentication | identity | compliance | document_format */
    @Column(name = "clause_type", length = 30)
    private String clauseType;

    @Column(name = "blocks_payment_reason", columnDefinition = "TEXT")
    private String blocksPaymentReason;
}
