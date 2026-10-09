package com.swiftvalidator.entity;

import jakarta.persistence.*;
import lombok.*;

import java.math.BigDecimal;

@Entity
@Table(name = "nlp_clauses")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NlpClause {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "analysis_id", nullable = false)
    private SwiftAnalysis analysis;

    @Column(name = "text", columnDefinition = "TEXT")
    private String text;

    @Column(name = "clause_type", length = 60)
    private String clauseType;

    @Column(name = "importance", length = 20)
    private String importance;

    @Column(name = "risk_level", length = 20)
    private String riskLevel;

    @Column(name = "confidence", precision = 4, scale = 3)
    private BigDecimal confidence;

    @Column(name = "method", length = 20)
    private String method;

    @Column(name = "ucp_refs", length = 200)
    private String ucpRefs;
}
