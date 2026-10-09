package com.swiftvalidator.dto;

import lombok.*;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LcCrossValidationResponseDTO {

    private Long       swiftAnalysisId;
    private Long       commercialValidationId;
    private BigDecimal finalScore;
    private String     riskLevel;
    private String     decision;               // ACCEPT | DISCREPANT | REJECT (LC-Commercial v3.0)
    /** UCP 600 decision: ACCEPT | ACCEPT_WITH_FLAGS | REVIEW | CONDITIONAL_HOLD | REJECT */
    private String     ucpDecision;
    private Boolean    contractualHold;

    private Map<String, BigDecimal> scoreBreakdown;
    private Map<String, BigDecimal> weights;
    private Map<String, BigDecimal> categoryWeights;  // v3.0 category weights

    private Map<String, Object>       categories;     // v3.0 per-category results
    private Map<String, Object>       crossDocument;  // v3.0 cross-document reconciliation
    private Map<String, Object>       xai;            // v3.0 Shapley XAI engine output

    private List<Map<String, Object>> ruleResults;
    private List<Map<String, Object>> fuzzyResults;
    private Map<String, Object>       semanticResults;
    private List<Map<String, Object>> conditionResults;
    private List<Map<String, Object>> explanations;

    private LcSummaryDTO         lcSummary;
    private CommercialSummaryInfo commercialInfo;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LcSummaryDTO {
        private String     lcNumber;
        private String     beneficiary;
        private String     applicant;
        private String     currency;
        private BigDecimal amount;
        private Integer    tolerancePlus;
        private Integer    toleranceMinus;
        private String     expiryDate;
        private String     shipmentDate;
        private Integer    presentationPeriodDays;
        private String     incoterm;
        private Boolean    partialShipmentsAllowed;
        private Boolean    transhipmentAllowed;
        private String     goodsDescription;
        private String     additionalConditions;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CommercialSummaryInfo {
        private Integer    nDocuments;
        private List<String> docTypes;
        private String     riskLevel;
        private BigDecimal crossScore;
    }
}
