package com.swiftvalidator.dto;

import lombok.Data;
import lombok.Builder;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Data @Builder @NoArgsConstructor @AllArgsConstructor
public class AnalysisResponseDTO {

    private Long id;
    private String lcNumber;
    private String swiftType;
    private String lcForm;

    private String applicant;
    private String beneficiary;
    private String issuingBank;

    private String currency;
    private BigDecimal amount;
    private Integer tolerancePlus;
    private Integer toleranceMinus;

    private LocalDate issueDate;
    private LocalDate expiryDate;
    private LocalDate shipmentDate;
    private Integer presentationPeriodDays;
    private Integer validityDays;

    private String incoterm;
    private String country;
    private Boolean partialShipmentsAllowed;
    private Boolean transhipmentAllowed;
    private String confirmationInstructions;
    private String applicableRules;
    private String goodsDescription;
    private String documentsRequired;
    private String additionalConditions;

    private Integer ucpComplianceScore;
    private Integer nlpScore;
    private Integer completenessScore;
    private Integer semanticScore;

    private String riskLevel;
    private BigDecimal riskScore;       // DS score — échelle 0-1 (legacy)
    private BigDecimal mlRiskScore;     // DS ML score — échelle 0-1 (legacy)
    /** Score de risque final unifié 0-100 (max des sources : ML, DS, UCP, analytique). */
    private Integer finalRiskScore;
    private Boolean paymentBlocked;
    private String countryRisk;
    private List<String> riskFactors;

    private Integer nUcpErrors;
    private Integer nUcpWarnings;
    private Integer nUcpExclusions;
    private Integer nClauses;
    private Integer nCriticalClauses;

    private List<UcpFindingDTO> ucpFindings;
    private List<NlpClauseDTO> nlpClauses;
    private ExternalValidationDTO externalValidation;
    private DsRiskScoreDTO dsRiskScore;
    private RiskMlDTO riskMl;
    private SwiftValidationDTO swiftValidation;
    private SemanticValidationDTO semanticValidation;

    private String status;
    private String validationType;   // MT700_ONLY | FULL_VALIDATION
    private Long processingMs;
    private LocalDateTime createdAt;
    private String rawPayloadJson;

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class UcpFindingDTO {
        private String article;
        private String articleTitle;
        private String code;
        private String category;
        private String severity;
        private Integer tier;
        private BigDecimal weight;
        private BigDecimal confidence;
        private String message;
        private String clauseText;
        private String recommendation;
        private Boolean blocksPayment;
        private Integer cascadeImpact;
        private String exclusionType;
        /** true_violation | contractual_modification | operational_discrepancy */
        private String findingNature;
        /** HIGH | MEDIUM | LOW */
        private String impactLevel;
        /** financial | transport | authentication | identity | compliance | document_format */
        private String clauseType;
        private String blocksPaymentReason;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class NlpClauseDTO {
        private String text;
        private String clauseType;
        private String importance;
        private String riskLevel;
        private BigDecimal confidence;
        private String method;
        private String ucpRefs;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class ExternalValidationDTO {
        private String riskLevel;
        private BigDecimal rawScore;
        private String approach;
        private String currencyCode;
        private Boolean currencyValid;
        private String countryIso2;
        private Boolean countrySanctioned;
        private String fatfList;
        private String sourcesConsulted;
        private Long processingMs;
        private ExternalDetailsDTO details;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class DsRiskScoreDTO {
        private String riskLevel;
        private BigDecimal riskScore;
        private BigDecimal mlRiskScore;
        private String modelUsed;
        private Integer completenessScore;
        private Integer nClauses;
        private Integer nCriticalClauses;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class SwiftValidationDTO {
        private Integer score;
        private Boolean overallValid;
        private String summary;
        private List<String> missingMandatory;
        private List<SwiftFieldValidationDTO> fields;
        private Integer nErrors;
        private Integer nWarnings;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class SwiftFieldValidationDTO {
        private String tag;
        private String name;
        private Boolean mandatory;
        private Boolean valid;
        private String value;
        private List<String> errors;
        private List<String> warnings;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class ExternalDetailsDTO {
        private List<ExternalCountryDTO> countries;
        private ExternalCurrencyDTO currency;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class ExternalCountryDTO {
        private String raw;
        private String iso2;
        private String risk;
        private String details;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class ExternalCurrencyDTO {
        private String code;
        private Boolean isValid;
        private String risk;
        private String details;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class RiskMlDTO {
        private String riskLabel;
        private Map<String, BigDecimal> riskProba;
        private Integer riskScore;
        private Boolean isAnomaly;
        private BigDecimal anomalyScore;
        private String segment;
        private Integer segmentId;
        private String narrative;
        private List<ShapFeatureDTO> shapTopFeatures;
        private List<UcpViolationDTO> ucpViolations;
        private String modelVersion;
        private String challengerLabel;
        private Long processingMs;
        private String error;
        // v4.0 additions
        private ScoreBreakdownDTO scoreBreakdown;
        private List<MissingFieldItemDTO> missingFieldsDetail;
        private ExplanationDTO explanation;
        private UcpDetailDTO ucpDetail;
        private Integer finalRiskScore;
        private Map<String, Integer> finalRiskScoreComponents;
    }

    // ── Score Breakdown ──────────────────────────────────────────────────────
    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class ScoreBreakdownDTO {
        private Double analyticalScore;
        private String analyticalLabel;
        private Map<String, Double> weights;
        private Map<String, Object> dimensions;
    }

    // ── Missing Fields ───────────────────────────────────────────────────────
    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class MissingFieldItemDTO {
        private String field;
        private String name;
        private String criticality;
        private Integer impactScore;
    }

    // ── SHAP Waterfall ───────────────────────────────────────────────────────
    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class ShapWaterfallItemDTO {
        private String feature;
        private BigDecimal value;
        private BigDecimal shapValue;
        private BigDecimal contributionPct;
        private String direction;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class RuleContributionDTO {
        private String rule;
        private String field;
        private String severity;
        private String description;
        private String scoreImpact;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class TopDimensionDTO {
        private String dimension;
        private Double contribution;
    }

    // ── Explanation (SHAP waterfall + radar) ─────────────────────────────────
    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class ExplanationDTO {
        private String method;
        private String finalLabel;
        private Integer riskScoreMl;
        private Double analyticalScore;
        private List<ShapWaterfallItemDTO> shapWaterfall;
        private Map<String, Double> dimensionRadar;
        private List<RuleContributionDTO> ruleContributions;
        private List<TopDimensionDTO> topContributingDimensions;
    }

    // ── UCP Detail (errors vs warnings séparés) ───────────────────────────────
    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class UcpDetailDTO {
        private Integer errorsCount;
        private Integer warningsCount;
        private Double weightedViolationIndex;
        private Double errorWeight;
        private Double warningWeight;
        private Double complianceScorePct;
        private String ucpDecision;
        private Boolean paymentBlocked;
        private Integer ruleViolationsCount;
        private List<Map<String, Object>> blockViolations;
        private List<Map<String, Object>> highViolations;
        private List<Map<String, Object>> warnViolations;
        private List<Object> errorsDetail;
        private List<Object> warningsDetail;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class UcpViolationDTO {
        private String article;
        private String field;
        private String description;
        private String severity;
        private String overrideScore;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class ShapFeatureDTO {
        private String feature;
        private BigDecimal shapValue;
        private String direction;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class SemanticValidationDTO {
        private Integer completenessScore;
        private Integer semanticScore;
        private BigDecimal bayesianConfidence;
        private Integer docQualityScore;
        private String summary;
        private List<SemanticItemDTO> warnings;
        private List<SemanticItemDTO> contradictions;
        private List<SemanticItemDTO> riskFlags;
        private List<String> recommendations;
        private String coherenceScoresJson;
        private String geoCoherenceJson;
        private List<String> fraudIndicators;

        // AXE 7 — Cross-field coherence (NLI + rules)
        private List<CrossFieldIssueDTO> crossFieldIssues;
        private Double crossFieldCoherenceScore;
        private String crossFieldVerdict;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class CrossFieldIssueDTO {
        private List<String> fields;
        private String code;
        private String technique;   // "NLI" | "NLI+regex" | "regex" | "rule"
        private String severity;    // "error" | "warning" | "info"
        private String message;
        private String suggestion;
        private String lcValue;
        private String docValue;
        private Double confidence;
        private String ucpRef;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class SemanticItemDTO {
        private String code;
        private String severity;
        private String message;
        private String detail;
        private String field;
        private String relatedField;
    }
}
