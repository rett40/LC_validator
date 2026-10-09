package com.swiftvalidator.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "swift_analyses")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SwiftAnalysis {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "lc_number", length = 50)
    private String lcNumber;

    @Column(name = "swift_type", length = 10)
    private String swiftType;

    @Column(name = "lc_form", length = 30)
    private String lcForm;

    @Column(name = "applicant", length = 500)
    private String applicant;

    @Column(name = "beneficiary", length = 500)
    private String beneficiary;

    @Column(name = "issuing_bank", length = 200)
    private String issuingBank;

    @Column(name = "currency", length = 3)
    private String currency;

    @Column(name = "amount", precision = 20, scale = 2)
    private BigDecimal amount;

    @Column(name = "tolerance_plus")
    private Integer tolerancePlus;

    @Column(name = "tolerance_minus")
    private Integer toleranceMinus;

    @Column(name = "issue_date")
    private LocalDate issueDate;

    @Column(name = "expiry_date")
    private LocalDate expiryDate;

    @Column(name = "expiry_place", length = 100)
    private String expiryPlace;

    @Column(name = "shipment_date")
    private LocalDate shipmentDate;

    @Column(name = "presentation_period_days")
    private Integer presentationPeriodDays;

    @Column(name = "validity_days")
    private Integer validityDays;

    @Column(name = "incoterm", length = 10)
    private String incoterm;

    @Column(name = "country", length = 100)
    private String country;

    @Column(name = "partial_shipments_allowed")
    private Boolean partialShipmentsAllowed;

    @Column(name = "transhipment_allowed")
    private Boolean transhipmentAllowed;

    @Column(name = "confirmation_instructions", length = 50)
    private String confirmationInstructions;

    @Column(name = "applicable_rules", length = 100)
    private String applicableRules;

    @Column(name = "goods_description", columnDefinition = "TEXT")
    private String goodsDescription;

    @Column(name = "documents_required", columnDefinition = "TEXT")
    private String documentsRequired;

    @Column(name = "additional_conditions", columnDefinition = "TEXT")
    private String additionalConditions;

    @Column(name = "charges_details", columnDefinition = "TEXT")
    private String chargesDetails;

    @Column(name = "charges_beneficiary")
    private Boolean chargesBeneficiary;

    @Column(name = "ucp_compliance_score")
    private Integer ucpComplianceScore;

    @Column(name = "nlp_score")
    private Integer nlpScore;

    @Column(name = "completeness_score")
    private Integer completenessScore;

    @Column(name = "semantic_score")
    private Integer semanticScore;

    @Column(name = "validation_score")
    private Integer validationScore;

    @Column(name = "risk_level", length = 20)
    private String riskLevel;

    @Column(name = "risk_score", precision = 6, scale = 4)
    private BigDecimal riskScore;

    @Column(name = "ml_risk_score", precision = 6, scale = 4)
    private BigDecimal mlRiskScore;

    @Column(name = "incoterm_risk_score")
    private Integer incotermRiskScore;

    @Column(name = "country_risk", length = 20)
    private String countryRisk;

    @Column(name = "payment_blocked")
    private Boolean paymentBlocked;

    @Column(name = "n_ucp_errors")
    private Integer nUcpErrors;

    @Column(name = "n_ucp_warnings")
    private Integer nUcpWarnings;

    @Column(name = "n_ucp_exclusions")
    private Integer nUcpExclusions;

    @Column(name = "n_clauses")
    private Integer nClauses;

    @Column(name = "n_critical_clauses")
    private Integer nCriticalClauses;

    @Column(name = "n_high_risk_clauses")
    private Integer nHighRiskClauses;

    @Column(name = "raw_text", columnDefinition = "TEXT")
    private String rawText;

    @Column(name = "ocr_confidence", precision = 4, scale = 3)
    private BigDecimal ocrConfidence;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "full_payload", columnDefinition = "jsonb")
    private String fullPayload;

    @Column(name = "status", length = 20)
    @Builder.Default
    private String status = "PENDING";

    /** MT700_ONLY | FULL_VALIDATION */
    @Column(name = "validation_type", length = 20)
    @Builder.Default
    private String validationType = "MT700_ONLY";

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @Column(name = "processing_ms")
    private Long processingMs;

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    @OneToOne(cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    @JoinColumn(name = "document_id")
    private Document document;

    @OneToMany(mappedBy = "analysis", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<UcpFinding> ucpFindings = new ArrayList<>();

    @OneToMany(mappedBy = "analysis", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<NlpClause> nlpClauses = new ArrayList<>();

    @OneToOne(mappedBy = "analysis", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private ExternalValidation externalValidation;

    @OneToOne(mappedBy = "analysis", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private DsRiskScore dsRiskScore;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
