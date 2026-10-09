package com.swiftvalidator.mapper;

import com.swiftvalidator.dto.AnalysisResponseDTO;
import com.swiftvalidator.entity.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

@Component
public class AnalysisMapper {
    private static final String SCORE_FIELD = "score";

    private final ObjectMapper objectMapper;

    public AnalysisMapper(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public AnalysisResponseDTO toDTO(SwiftAnalysis a) {
        return toDTO(a, true);
    }

    public AnalysisResponseDTO toDTO(SwiftAnalysis a, boolean includeRawPayload) {
        if (a == null) return null;

        return AnalysisResponseDTO.builder()
            .id(a.getId())
            .lcNumber(a.getLcNumber())
            .swiftType(a.getSwiftType())
            .lcForm(a.getLcForm())
            .applicant(a.getApplicant())
            .beneficiary(a.getBeneficiary())
            .issuingBank(a.getIssuingBank())
            .currency(a.getCurrency())
            .amount(a.getAmount())
            .tolerancePlus(a.getTolerancePlus())
            .toleranceMinus(a.getToleranceMinus())
            .issueDate(a.getIssueDate())
            .expiryDate(a.getExpiryDate())
            .shipmentDate(a.getShipmentDate())
            .presentationPeriodDays(a.getPresentationPeriodDays())
            .validityDays(a.getValidityDays())
            .incoterm(a.getIncoterm())
            .country(a.getCountry())
            .partialShipmentsAllowed(a.getPartialShipmentsAllowed())
            .transhipmentAllowed(a.getTranshipmentAllowed())
            .confirmationInstructions(a.getConfirmationInstructions())
            .applicableRules(a.getApplicableRules())
            .goodsDescription(a.getGoodsDescription())
            .documentsRequired(a.getDocumentsRequired())
            .additionalConditions(a.getAdditionalConditions())
            .ucpComplianceScore(a.getUcpComplianceScore())
            .nlpScore(a.getNlpScore())
            .completenessScore(a.getCompletenessScore())
            .semanticScore(a.getSemanticScore())
            .riskLevel(a.getRiskLevel())
            .riskScore(a.getRiskScore())
            .mlRiskScore(a.getMlRiskScore())
            .finalRiskScore(extractFinalRiskScore(a.getFullPayload(), a.getRiskScore()))
            .paymentBlocked(a.getPaymentBlocked())
            .countryRisk(a.getCountryRisk())
            .riskFactors(extractRiskFactors(a.getFullPayload()))
            .nUcpErrors(a.getNUcpErrors())
            .nUcpWarnings(a.getNUcpWarnings())
            .nUcpExclusions(a.getNUcpExclusions())
            .nClauses(a.getNClauses())
            .nCriticalClauses(a.getNCriticalClauses())
            .ucpFindings(a.getUcpFindings().stream().map(this::toUcpFindingDTO).toList())
            .nlpClauses(a.getNlpClauses().stream().map(this::toNlpClauseDTO).toList())
            .externalValidation(toExternalValidationDTO(a.getExternalValidation()))
            .dsRiskScore(toDsRiskScoreDTO(a.getDsRiskScore()))
            .riskMl(extractRiskMl(a.getFullPayload()))
            .swiftValidation(extractSwiftValidation(a.getFullPayload()))
            .semanticValidation(extractSemanticValidation(a.getFullPayload()))
            .status(a.getStatus())
            .validationType(a.getValidationType())
            .processingMs(a.getProcessingMs())
            .createdAt(a.getCreatedAt())
            .rawPayloadJson(includeRawPayload ? a.getFullPayload() : null)
            .build();
    }

    private List<String> extractRiskFactors(String fullPayload) {
        if (fullPayload == null || fullPayload.isBlank()) return List.of();
        try {
            JsonNode root = objectMapper.readTree(fullPayload);
            return toStringList(root.path("ds_analysis").path("risk").path("risk_factors"));
        } catch (Exception e) {
            return List.of();
        }
    }

    private AnalysisResponseDTO.UcpFindingDTO toUcpFindingDTO(UcpFinding f) {
        if (f == null) return null;
        return AnalysisResponseDTO.UcpFindingDTO.builder()
            .article(f.getArticle())
            .articleTitle(f.getArticleTitle())
            .code(f.getCode())
            .category(f.getCategory())
            .severity(f.getSeverity())
            .tier(f.getTier())
            .weight(f.getWeight())
            .confidence(f.getConfidence())
            .message(f.getMessage())
            .clauseText(f.getClauseText())
            .recommendation(f.getRecommendation())
            .blocksPayment(f.getBlocksPayment())
            .cascadeImpact(f.getCascadeImpact())
            .exclusionType(f.getExclusionType())
            .findingNature(f.getFindingNature())
            .impactLevel(f.getImpactLevel())
            .clauseType(f.getClauseType())
            .blocksPaymentReason(f.getBlocksPaymentReason())
            .build();
    }

    private AnalysisResponseDTO.NlpClauseDTO toNlpClauseDTO(NlpClause c) {
        if (c == null) return null;
        return AnalysisResponseDTO.NlpClauseDTO.builder()
            .text(c.getText())
            .clauseType(c.getClauseType())
            .importance(c.getImportance())
            .riskLevel(c.getRiskLevel())
            .confidence(c.getConfidence())
            .method(c.getMethod())
            .ucpRefs(c.getUcpRefs())
            .build();
    }

    private AnalysisResponseDTO.ExternalValidationDTO toExternalValidationDTO(ExternalValidation e) {
        if (e == null) return null;
        return AnalysisResponseDTO.ExternalValidationDTO.builder()
            .riskLevel(e.getRiskLevel())
            .rawScore(e.getRawScore())
            .approach(e.getApproach())
            .currencyCode(e.getCurrencyCode())
            .currencyValid(e.getCurrencyValid())
            .countryIso2(e.getCountryIso2())
            .countrySanctioned(e.getCountrySanctioned())
            .fatfList(e.getFatfList())
            .sourcesConsulted(e.getSourcesConsulted())
            .processingMs(e.getProcessingMs())
            .details(extractExternalDetailsFromPayload(e.getDetailsJson()))
            .build();
    }

    private AnalysisResponseDTO.DsRiskScoreDTO toDsRiskScoreDTO(DsRiskScore d) {
        if (d == null) return null;
        return AnalysisResponseDTO.DsRiskScoreDTO.builder()
            .riskLevel(d.getRiskLevel())
            .riskScore(d.getRiskScore())
            .mlRiskScore(d.getMlRiskScore())
            .modelUsed(d.getModelUsed())
            .completenessScore(d.getCompletenessScore())
            .nClauses(d.getNClauses())
            .nCriticalClauses(d.getNCriticalClauses())
            .build();
    }

    private AnalysisResponseDTO.RiskMlDTO extractRiskMl(String fullPayload) {
        if (fullPayload == null || fullPayload.isBlank()) return null;
        try {
            JsonNode root = objectMapper.readTree(fullPayload);
            JsonNode riskMl = root.path("risk_ml");
            if (riskMl.isMissingNode() || riskMl.isNull() || riskMl.isEmpty()) return null;

            List<AnalysisResponseDTO.ShapFeatureDTO> shap = new ArrayList<>();
            for (JsonNode f : riskMl.path("shap_top_features")) {
                if (f == null || f.isNull()) continue;
                shap.add(AnalysisResponseDTO.ShapFeatureDTO.builder()
                        .feature(text(f, "feature"))
                        .shapValue(decimalFromAny(f.get("shap_value")))
                        .direction(text(f, "direction"))
                        .build());
            }

            Map<String, BigDecimal> riskProbaMap = new HashMap<>();
            JsonNode probaNode = riskMl.get("risk_proba");
            if (probaNode != null && probaNode.isObject()) {
                Iterator<Map.Entry<String, JsonNode>> fields = probaNode.fields();
                while (fields.hasNext()) {
                    Map.Entry<String, JsonNode> entry = fields.next();
                    BigDecimal val = decimalFromAny(entry.getValue());
                    if (val != null) riskProbaMap.put(entry.getKey(), val);
                }
            }

            List<AnalysisResponseDTO.UcpViolationDTO> ucpViolations = new ArrayList<>();
            for (JsonNode v : riskMl.path("ucp_violations")) {
                if (v == null || v.isNull()) continue;
                ucpViolations.add(AnalysisResponseDTO.UcpViolationDTO.builder()
                        .article(text(v, "article"))
                        .field(text(v, "field"))
                        .description(text(v, "description"))
                        .severity(text(v, "severity"))
                        .overrideScore(text(v, "override_score"))
                        .build());
            }

            return AnalysisResponseDTO.RiskMlDTO.builder()
                    .riskLabel(text(riskMl, "risk_label"))
                    .riskProba(riskProbaMap.isEmpty() ? null : riskProbaMap)
                    .riskScore(intFromAny(riskMl.get("risk_score")))
                    .isAnomaly(boolFromAny(riskMl.get("is_anomaly")))
                    .anomalyScore(decimalFromAny(riskMl.get("anomaly_score")))
                    .segment(text(riskMl, "segment"))
                    .segmentId(intFromAny(riskMl.get("segment_id")))
                    .narrative(text(riskMl, "narrative"))
                    .shapTopFeatures(shap)
                    .ucpViolations(ucpViolations.isEmpty() ? null : ucpViolations)
                    .modelVersion(text(riskMl, "model_version"))
                    .challengerLabel(text(riskMl, "challenger_label"))
                    .processingMs(longFromAny(riskMl.get("processing_ms")))
                    .error(text(riskMl, "error"))
                    // v4.0 additions
                    .scoreBreakdown(extractScoreBreakdown(riskMl.path("score_breakdown")))
                    .missingFieldsDetail(extractMissingFieldsDetail(riskMl.path("missing_fields_detail")))
                    .explanation(extractExplanation(riskMl.path("explanation")))
                    .ucpDetail(extractUcpDetail(riskMl.path("ucp_detail")))
                    .finalRiskScore(intFromAny(riskMl.get("final_risk_score")))
                    .finalRiskScoreComponents(extractIntMap(riskMl.path("final_risk_score_components")))
                    .build();
        } catch (Exception e) {
            return null;
        }
    }

    private AnalysisResponseDTO.ScoreBreakdownDTO extractScoreBreakdown(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) return null;
        try {
            Map<String, Double> weights = new HashMap<>();
            JsonNode wNode = node.path("weights");
            if (wNode.isObject()) {
                wNode.fields().forEachRemaining(e -> {
                    Double v = doubleFromAny(e.getValue());
                    if (v != null) weights.put(e.getKey(), v);
                });
            }
            Map<String, Object> dims = new HashMap<>();
            JsonNode dimsNode = node.path("dimensions");
            if (dimsNode.isObject()) {
                dimsNode.fields().forEachRemaining(e -> {
                    try {
                        dims.put(e.getKey(), objectMapper.convertValue(e.getValue(), Map.class));
                    } catch (Exception ex) {
                        dims.put(e.getKey(), e.getValue().toString());
                    }
                });
            }
            return AnalysisResponseDTO.ScoreBreakdownDTO.builder()
                    .analyticalScore(doubleFromAny(node.get("analytical_score")))
                    .analyticalLabel(text(node, "analytical_label"))
                    .weights(weights.isEmpty() ? null : weights)
                    .dimensions(dims.isEmpty() ? null : dims)
                    .build();
        } catch (Exception e) {
            return null;
        }
    }

    private List<AnalysisResponseDTO.MissingFieldItemDTO> extractMissingFieldsDetail(JsonNode node) {
        List<AnalysisResponseDTO.MissingFieldItemDTO> out = new ArrayList<>();
        if (node == null || node.isMissingNode() || node.isNull() || !node.isArray()) return out;
        for (JsonNode item : node) {
            if (item == null || item.isNull()) continue;
            out.add(AnalysisResponseDTO.MissingFieldItemDTO.builder()
                    .field(text(item, "field"))
                    .name(text(item, "name"))
                    .criticality(text(item, "criticality"))
                    .impactScore(intFromAny(item.get("impact_score")))
                    .build());
        }
        return out;
    }

    private AnalysisResponseDTO.ExplanationDTO extractExplanation(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) return null;
        try {
            List<AnalysisResponseDTO.ShapWaterfallItemDTO> waterfall = new ArrayList<>();
            for (JsonNode w : node.path("shap_waterfall")) {
                if (w == null || w.isNull()) continue;
                waterfall.add(AnalysisResponseDTO.ShapWaterfallItemDTO.builder()
                        .feature(text(w, "feature"))
                        .value(decimalFromAny(w.get("value")))
                        .shapValue(decimalFromAny(w.get("shap_value")))
                        .contributionPct(decimalFromAny(w.get("contribution_pct")))
                        .direction(text(w, "direction"))
                        .build());
            }
            Map<String, Double> radar = new HashMap<>();
            JsonNode radarNode = node.path("dimension_radar");
            if (radarNode.isObject()) {
                radarNode.fields().forEachRemaining(e -> {
                    Double v = doubleFromAny(e.getValue());
                    if (v != null) radar.put(e.getKey(), v);
                });
            }
            List<AnalysisResponseDTO.RuleContributionDTO> rules = new ArrayList<>();
            for (JsonNode r : node.path("rule_contributions")) {
                if (r == null || r.isNull()) continue;
                rules.add(AnalysisResponseDTO.RuleContributionDTO.builder()
                        .rule(text(r, "rule"))
                        .field(text(r, "field"))
                        .severity(text(r, "severity"))
                        .description(text(r, "description"))
                        .scoreImpact(text(r, "score_impact"))
                        .build());
            }
            List<AnalysisResponseDTO.TopDimensionDTO> topDims = new ArrayList<>();
            for (JsonNode td : node.path("top_contributing_dimensions")) {
                if (td == null || td.isNull()) continue;
                topDims.add(AnalysisResponseDTO.TopDimensionDTO.builder()
                        .dimension(text(td, "dimension"))
                        .contribution(doubleFromAny(td.get("contribution")))
                        .build());
            }
            return AnalysisResponseDTO.ExplanationDTO.builder()
                    .method(text(node, "method"))
                    .finalLabel(text(node, "final_label"))
                    .riskScoreMl(intFromAny(node.get("risk_score_ml")))
                    .analyticalScore(doubleFromAny(node.get("analytical_score")))
                    .shapWaterfall(waterfall.isEmpty() ? null : waterfall)
                    .dimensionRadar(radar.isEmpty() ? null : radar)
                    .ruleContributions(rules.isEmpty() ? null : rules)
                    .topContributingDimensions(topDims.isEmpty() ? null : topDims)
                    .build();
        } catch (Exception e) {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private AnalysisResponseDTO.UcpDetailDTO extractUcpDetail(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull()) return null;
        try {
            List<Map<String, Object>> blockViol = new ArrayList<>();
            List<Map<String, Object>> highViol  = new ArrayList<>();
            List<Map<String, Object>> warnViol  = new ArrayList<>();
            for (JsonNode v : node.path("block_violations")) {
                if (v != null && !v.isNull()) blockViol.add(objectMapper.convertValue(v, Map.class));
            }
            for (JsonNode v : node.path("high_violations")) {
                if (v != null && !v.isNull()) highViol.add(objectMapper.convertValue(v, Map.class));
            }
            for (JsonNode v : node.path("warn_violations")) {
                if (v != null && !v.isNull()) warnViol.add(objectMapper.convertValue(v, Map.class));
            }
            List<Object> errorsDetail   = new ArrayList<>();
            List<Object> warningsDetail = new ArrayList<>();
            for (JsonNode e : node.path("errors_detail")) {
                if (e != null && !e.isNull()) {
                    try { errorsDetail.add(objectMapper.convertValue(e, Map.class)); }
                    catch (Exception ex) { errorsDetail.add(e.asText()); }
                }
            }
            for (JsonNode w : node.path("warnings_detail")) {
                if (w != null && !w.isNull()) {
                    try { warningsDetail.add(objectMapper.convertValue(w, Map.class)); }
                    catch (Exception ex) { warningsDetail.add(w.asText()); }
                }
            }
            return AnalysisResponseDTO.UcpDetailDTO.builder()
                    .errorsCount(intFromAny(node.get("errors_count")))
                    .warningsCount(intFromAny(node.get("warnings_count")))
                    .weightedViolationIndex(doubleFromAny(node.get("weighted_violation_index")))
                    .errorWeight(doubleFromAny(node.get("error_weight")))
                    .warningWeight(doubleFromAny(node.get("warning_weight")))
                    .complianceScorePct(doubleFromAny(node.get("compliance_score_pct")))
                    .ucpDecision(text(node, "ucp_decision"))
                    .paymentBlocked(boolFromAny(node.get("payment_blocked")))
                    .ruleViolationsCount(intFromAny(node.get("rule_violations_count")))
                    .blockViolations(blockViol.isEmpty() ? null : blockViol)
                    .highViolations(highViol.isEmpty() ? null : highViol)
                    .warnViolations(warnViol.isEmpty() ? null : warnViol)
                    .errorsDetail(errorsDetail.isEmpty() ? null : errorsDetail)
                    .warningsDetail(warningsDetail.isEmpty() ? null : warningsDetail)
                    .build();
        } catch (Exception e) {
            return null;
        }
    }

    private Map<String, Integer> extractIntMap(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isNull() || !node.isObject()) return null;
        Map<String, Integer> out = new HashMap<>();
        node.fields().forEachRemaining(e -> {
            Integer v = intFromAny(e.getValue());
            if (v != null) out.put(e.getKey(), v);
        });
        return out.isEmpty() ? null : out;
    }

    private Double doubleFromAny(JsonNode val) {
        if (val == null || val.isNull() || val.isMissingNode()) return null;
        try { return val.asDouble(); } catch (Exception e) { return null; }
    }

    /**
     * Extrait le score final unifié 0-100 depuis risk_ml.final_risk_score.
     * Si absent, normalise le DS riskScore (0-1 → 0-100).
     * Garantit une échelle cohérente pour l'affichage frontend.
     */
    private Integer extractFinalRiskScore(String fullPayload, BigDecimal dsRiskScore) {
        if (fullPayload != null && !fullPayload.isBlank()) {
            try {
                JsonNode root   = objectMapper.readTree(fullPayload);
                JsonNode mlNode = root.path("risk_ml");
                if (!mlNode.isMissingNode() && !mlNode.isNull()) {
                    Integer fromMl = intFromAny(mlNode.get("final_risk_score"));
                    if (fromMl != null && fromMl >= 0) return Math.min(100, fromMl);
                    // Fallback : risk_score direct
                    Integer fromRiskScore = intFromAny(mlNode.get("risk_score"));
                    if (fromRiskScore != null && fromRiskScore >= 0) return Math.min(100, fromRiskScore);
                }
            } catch (Exception ignored) { /* silencieux */ }
        }
        // Dernier recours : normaliser dsRiskScore (0-1) → 0-100
        if (dsRiskScore != null) {
            double v = dsRiskScore.doubleValue();
            if (v <= 1.0) return (int) Math.round(v * 100);
            return (int) Math.min(100, Math.round(v));
        }
        return null;
    }

    private AnalysisResponseDTO.SwiftValidationDTO extractSwiftValidation(String fullPayload) {
        if (fullPayload == null || fullPayload.isBlank()) return null;
        try {
            JsonNode root = objectMapper.readTree(fullPayload);
            JsonNode v = root.path("swift").path("validation");
            if (v.isMissingNode() || v.isNull() || v.isEmpty()) return null;

            List<String> missingMandatory = new ArrayList<>();
            for (JsonNode t : v.path("missing_mandatory")) {
                if (t != null && !t.isNull()) missingMandatory.add(t.asText());
            }

            List<AnalysisResponseDTO.SwiftFieldValidationDTO> fields = extractSwiftFieldValidations(v.path("fields"), missingMandatory);
            int[] counts = countSwiftIssues(fields);
            int nErrors = counts[0];
            int nWarnings = counts[1];

            return AnalysisResponseDTO.SwiftValidationDTO.builder()
                    .score(intFromAny(v.get(SCORE_FIELD)))
                    .overallValid(boolFromAny(v.get("overall_valid")))
                    .summary(text(v, "summary"))
                    .missingMandatory(missingMandatory)
                    .fields(fields)
                    .nErrors(nErrors)
                    .nWarnings(nWarnings)
                    .build();
        } catch (Exception e) {
            return null;
        }
    }

    private List<AnalysisResponseDTO.SwiftFieldValidationDTO> extractSwiftFieldValidations(
            JsonNode fieldsNode,
            List<String> missingMandatory
    ) {
        List<AnalysisResponseDTO.SwiftFieldValidationDTO> fields = new ArrayList<>();
        if (fieldsNode == null || !fieldsNode.isObject()) return fields;

        fieldsNode.fields().forEachRemaining(entry -> {
            String tagKey = entry.getKey();
            JsonNode fv = entry.getValue();
            List<String> errors = toStringList(fv.path("errors"));
            List<String> warns = toStringList(fv.path("warnings"));
            boolean mandatory = errors.stream().anyMatch(s -> s.toLowerCase().contains("obligatoire")) ||
                    missingMandatory.contains(tagKey);

            fields.add(AnalysisResponseDTO.SwiftFieldValidationDTO.builder()
                    .tag(text(fv, "tag") != null ? text(fv, "tag") : tagKey)
                    .name(text(fv, "name"))
                    .mandatory(mandatory)
                    .valid(boolFromAny(fv.get("valid")))
                    .value(stringifyValue(fv.get("value")))
                    .errors(errors)
                    .warnings(warns)
                    .build());
        });
        return fields;
    }

    private int[] countSwiftIssues(List<AnalysisResponseDTO.SwiftFieldValidationDTO> fields) {
        int nErrors = 0;
        int nWarnings = 0;
        for (AnalysisResponseDTO.SwiftFieldValidationDTO f : fields) {
            nErrors += f.getErrors() == null ? 0 : f.getErrors().size();
            nWarnings += f.getWarnings() == null ? 0 : f.getWarnings().size();
        }
        return new int[] { nErrors, nWarnings };
    }

    private AnalysisResponseDTO.ExternalDetailsDTO extractExternalDetailsFromPayload(String detailsJson) {
        if (detailsJson == null || detailsJson.isBlank()) return null;
        try {
            JsonNode d = objectMapper.readTree(detailsJson);
            List<AnalysisResponseDTO.ExternalCountryDTO> countries = new ArrayList<>();
            for (JsonNode c : d.path("countries")) {
                if (c == null || c.isNull()) continue;
                countries.add(AnalysisResponseDTO.ExternalCountryDTO.builder()
                        .raw(text(c, "raw"))
                        .iso2(text(c, "iso2"))
                        .risk(text(c, "risk"))
                        .details(text(c, "details"))
                        .build());
            }

            JsonNode cur = d.path("currency");
            AnalysisResponseDTO.ExternalCurrencyDTO currency = null;
            if (cur != null && !cur.isNull() && !cur.isMissingNode()) {
                currency = AnalysisResponseDTO.ExternalCurrencyDTO.builder()
                        .code(text(cur, "code"))
                        .isValid(boolFromAny(cur.get("is_valid")))
                        .risk(text(cur, "risk"))
                        .details(text(cur, "details"))
                        .build();
            }

            return AnalysisResponseDTO.ExternalDetailsDTO.builder()
                    .countries(countries)
                    .currency(currency)
                    .build();
        } catch (Exception e) {
            return null;
        }
    }

    private AnalysisResponseDTO.SemanticValidationDTO extractSemanticValidation(String fullPayload) {
        if (fullPayload == null || fullPayload.isBlank()) return null;
        try {
            JsonNode root = objectMapper.readTree(fullPayload);
            JsonNode sem = root.path("swift").path("nlp").path("semantic");
            if (sem.isMissingNode() || sem.isNull() || sem.isEmpty()) return null;

            JsonNode cfNode = sem.path("cross_field_coherence");
            List<AnalysisResponseDTO.CrossFieldIssueDTO> crossFieldIssues = extractCrossFieldIssues(cfNode.path("issues"));
            Double crossFieldScore = cfNode.isMissingNode() ? null : cfNode.path("coherence_score").asDouble(100.0);
            String crossFieldVerdict = cfNode.isMissingNode() ? null : text(cfNode, "verdict");

            return AnalysisResponseDTO.SemanticValidationDTO.builder()
                    .completenessScore(intFromAny(sem.get("completeness_score")))
                    .semanticScore(intFromAny(sem.get("semantic_score")))
                    .bayesianConfidence(decimalFromAny(sem.get("bayesian_confidence")))
                    .docQualityScore(intFromAny(sem.get("doc_quality_score")))
                    .summary(text(sem, "summary"))
                    .warnings(extractSemanticItems(sem.path("warnings")))
                    .contradictions(extractSemanticItems(sem.path("contradictions")))
                    .riskFlags(extractSemanticItems(sem.path("risk_flags")))
                    .recommendations(toStringList(sem.path("recommendations")))
                    .coherenceScoresJson(stringifyValue(sem.get("coherence_scores")))
                    .geoCoherenceJson(stringifyValue(sem.get("geo_coherence")))
                    .fraudIndicators(toStringList(sem.path("fraud_indicators")))
                    .crossFieldIssues(crossFieldIssues)
                    .crossFieldCoherenceScore(crossFieldScore)
                    .crossFieldVerdict(crossFieldVerdict)
                    .build();
        } catch (Exception e) {
            return null;
        }
    }

    private List<AnalysisResponseDTO.CrossFieldIssueDTO> extractCrossFieldIssues(JsonNode arrayNode) {
        List<AnalysisResponseDTO.CrossFieldIssueDTO> out = new ArrayList<>();
        if (arrayNode == null || arrayNode.isNull() || arrayNode.isMissingNode() || !arrayNode.isArray()) return out;
        for (JsonNode n : arrayNode) {
            if (n == null || n.isNull()) continue;
            List<String> fields = toStringList(n.path("fields"));
            Double confidence = n.path("confidence").isMissingNode() ? null : n.path("confidence").asDouble();
            out.add(AnalysisResponseDTO.CrossFieldIssueDTO.builder()
                    .fields(fields)
                    .code(text(n, "code"))
                    .technique(text(n, "technique"))
                    .severity(text(n, "severity"))
                    .message(text(n, "message"))
                    .suggestion(text(n, "suggestion"))
                    .lcValue(text(n, "lc_value"))
                    .docValue(text(n, "doc_value"))
                    .confidence(confidence)
                    .ucpRef(text(n, "ucp_ref"))
                    .build());
        }
        return out;
    }

    private List<AnalysisResponseDTO.SemanticItemDTO> extractSemanticItems(JsonNode arrayNode) {
        List<AnalysisResponseDTO.SemanticItemDTO> out = new ArrayList<>();
        if (arrayNode == null || arrayNode.isNull() || arrayNode.isMissingNode() || !arrayNode.isArray()) return out;
        for (JsonNode n : arrayNode) {
            if (n == null || n.isNull()) continue;
            out.add(AnalysisResponseDTO.SemanticItemDTO.builder()
                    .code(text(n, "code"))
                    .severity(text(n, "severity"))
                    .message(text(n, "message"))
                    .detail(text(n, "detail"))
                    .field(text(n, "field"))
                    .relatedField(text(n, "related_field"))
                    .build());
        }
        return out;
    }

    private List<String> toStringList(JsonNode node) {
        List<String> out = new ArrayList<>();
        if (node == null || node.isNull() || node.isMissingNode()) return out;
        if (node.isArray()) {
            for (JsonNode n : node) {
                if (n != null && !n.isNull()) out.add(n.asText());
            }
        }
        return out;
    }

    private String stringifyValue(JsonNode val) {
        if (val == null || val.isNull() || val.isMissingNode()) return null;
        if (val.isTextual()) return val.asText();
        if (val.isNumber() || val.isBoolean()) return val.asText();
        try {
            return objectMapper.writeValueAsString(val);
        } catch (Exception e) {
            return val.toString();
        }
    }

    private String text(JsonNode node, String field) {
        if (node == null) return null;
        JsonNode val = node.get(field);
        return val == null || val.isNull() ? null : val.asText();
    }

    private BigDecimal decimalFromAny(JsonNode val) {
        if (val == null || val.isNull() || val.isMissingNode()) return null;
        try {
            if (val.isNumber()) return new BigDecimal(val.asText());
            if (val.isTextual()) return new BigDecimal(val.asText());
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    private Integer intFromAny(JsonNode val) {
        if (val == null || val.isNull() || val.isMissingNode()) return null;
        try { return val.asInt(); } catch (Exception e) { return null; }
    }

    private Long longFromAny(JsonNode val) {
        if (val == null || val.isNull() || val.isMissingNode()) return null;
        try { return val.asLong(); } catch (Exception e) { return null; }
    }

    private Boolean boolFromAny(JsonNode val) {
        if (val == null || val.isNull() || val.isMissingNode()) return false;
        if (val.isBoolean()) return val.asBoolean();
        String s = val.asText("").toUpperCase();
        return "TRUE".equals(s) || "YES".equals(s) || "1".equals(s);
    }
}
