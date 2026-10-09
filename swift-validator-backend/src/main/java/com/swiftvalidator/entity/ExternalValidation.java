package com.swiftvalidator.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "external_validations")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExternalValidation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "analysis_id", nullable = false)
    private SwiftAnalysis analysis;

    @Column(name = "risk_level", length = 20)
    private String riskLevel;

    @Column(name = "raw_score", precision = 6, scale = 4)
    private BigDecimal rawScore;

    @Column(name = "approach", length = 10)
    private String approach;

    @Column(name = "currency_code", length = 3)
    private String currencyCode;

    @Column(name = "currency_valid")
    private Boolean currencyValid;

    @Column(name = "country_iso2", length = 5)
    private String countryIso2;

    @Column(name = "country_sanctioned")
    private Boolean countrySanctioned;

    @Column(name = "ofac_entity_count")
    private Integer ofacEntityCount;

    @Column(name = "fatf_list", length = 30)
    private String fatfList;

    @Column(name = "sources_consulted", length = 500)
    private String sourcesConsulted;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "details_json", columnDefinition = "jsonb")
    private String detailsJson;

    @Column(name = "processing_ms")
    private Long processingMs;

    @Column(name = "processed_at")
    private LocalDateTime processedAt;
}
