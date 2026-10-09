package com.swiftvalidator.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Data @Builder @NoArgsConstructor @AllArgsConstructor
public class CommercialValidationResponseDTO {

    private Long id;
    private String lcReference;
    private Long swiftAnalysisId;
    private String status;
    private String riskLevel;
    private BigDecimal finalScore;
    private Integer nDocuments;
    private Integer nProcessed;
    private Integer nFailed;
    private BigDecimal processingTimeS;
    private LocalDateTime createdAt;

    private SummaryDTO summary;
    private List<DocumentResultDTO> documents;
    private List<Map<String, Object>> crossValidation;
    private CrossReportDTO crossReport;
    private String fullPayloadJson;

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class SummaryDTO {
        private Integer totalDocuments;
        private Integer processed;
        private Integer failed;
        private BigDecimal processingTimeS;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class DocumentResultDTO {
        private Long id;
        private String sourceFilename;
        private String status;
        private String docType;
        private String inputType;
        private BigDecimal confidence;
        private Boolean confident;
        private BigDecimal extractionScore;
        private Integer nFields;
        private Integer nPages;
        private Integer nWarnings;
        private Integer nRuleFails;
        private Map<String, Object> fields;
        private Map<String, Object> tables;
        private Map<String, String> extractionTiers;
        private List<Map<String, Object>> ruleResults;
        private RuleSummaryDTO ruleSummary;
        private BigDecimal processingTimeS;
        private List<String> warnings;
        private String error;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class RuleSummaryDTO {
        private Integer pass;
        private Integer fail;
        private Integer warning;
        private Integer corrected;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class CrossReportDTO {
        private BigDecimal finalScore;
        private String riskLevel;
        private Map<String, BigDecimal> scoreBreakdown;
        private List<Map<String, Object>> ruleResults;
        private Map<String, BigDecimal> semanticScores;
        private List<Map<String, Object>> nliResults;
        private List<Map<String, Object>> tableMatches;
        private List<Map<String, Object>> explanations;
        private List<Map<String, Object>> documents;
    }
}
