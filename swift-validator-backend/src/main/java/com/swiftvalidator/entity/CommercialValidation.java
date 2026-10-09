package com.swiftvalidator.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "commercial_validations")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CommercialValidation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "lc_reference", length = 50)
    private String lcReference;

    @Column(name = "swift_analysis_id")
    private Long swiftAnalysisId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    @Column(name = "status", length = 20)
    @Builder.Default
    private String status = "PENDING";

    @Column(name = "risk_level", length = 20)
    private String riskLevel;

    @Column(name = "final_score", precision = 6, scale = 4)
    private BigDecimal finalScore;

    @Column(name = "n_documents")
    @Builder.Default
    private Integer nDocuments = 0;

    @Column(name = "n_processed")
    @Builder.Default
    private Integer nProcessed = 0;

    @Column(name = "n_failed")
    @Builder.Default
    private Integer nFailed = 0;

    @Column(name = "processing_time_s", precision = 10, scale = 3)
    private BigDecimal processingTimeS;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "cross_report", columnDefinition = "jsonb")
    private String crossReport;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "full_payload", columnDefinition = "jsonb")
    private String fullPayload;

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @OneToMany(mappedBy = "validation", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<CommercialDocument> documents = new ArrayList<>();

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
