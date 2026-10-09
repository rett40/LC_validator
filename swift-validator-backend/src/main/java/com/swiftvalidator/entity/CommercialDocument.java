package com.swiftvalidator.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.math.BigDecimal;

@Entity
@Table(name = "commercial_documents")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CommercialDocument {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "validation_id")
    private CommercialValidation validation;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "document_id")
    private Document document;

    @Column(name = "source_filename", length = 255)
    private String sourceFilename;

    @Column(name = "doc_type", length = 30)
    private String docType;

    @Column(name = "input_type", length = 30)
    private String inputType;

    @Column(name = "confidence", precision = 6, scale = 4)
    private BigDecimal confidence;

    @Column(name = "extraction_score", precision = 6, scale = 4)
    private BigDecimal extractionScore;

    @Column(name = "n_fields")
    @Builder.Default
    private Integer nFields = 0;

    @Column(name = "n_pages")
    @Builder.Default
    private Integer nPages = 0;

    @Column(name = "n_warnings")
    @Builder.Default
    private Integer nWarnings = 0;

    @Column(name = "n_rule_fails")
    @Builder.Default
    private Integer nRuleFails = 0;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "fields_json", columnDefinition = "jsonb")
    private String fieldsJson;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "tables_json", columnDefinition = "jsonb")
    private String tablesJson;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "tiers_json", columnDefinition = "jsonb")
    private String tiersJson;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "rule_results_json", columnDefinition = "jsonb")
    private String ruleResultsJson;

    @Column(name = "status", length = 20)
    private String status;

    @Column(name = "processing_time_s", precision = 10, scale = 3)
    private BigDecimal processingTimeS;
}
