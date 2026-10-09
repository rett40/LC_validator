package com.swiftvalidator.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.swiftvalidator.dto.LcCrossValidationResponseDTO;
import com.swiftvalidator.entity.CommercialValidation;
import com.swiftvalidator.entity.SwiftAnalysis;
import com.swiftvalidator.entity.User;
import com.swiftvalidator.repository.CommercialValidationRepository;
import com.swiftvalidator.repository.SwiftAnalysisRepository;
import com.swiftvalidator.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class LcCrossValidationService {

    private final PipelineClientService          pipelineClient;
    private final SwiftAnalysisRepository        swiftAnalysisRepo;
    private final CommercialValidationRepository  commercialValidationRepo;
    private final UserRepository                 userRepo;
    private final ObjectMapper                   objectMapper;

    @Transactional(readOnly = true)
    public LcCrossValidationResponseDTO validate(
            Long   swiftAnalysisId,
            Long   commercialValidationId,
            String username
    ) throws Exception {

        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        SwiftAnalysis lc = swiftAnalysisRepo.findById(swiftAnalysisId)
                .orElseThrow(() -> new RuntimeException("SwiftAnalysis not found: " + swiftAnalysisId));

        CommercialValidation commercial = commercialValidationRepo
                .findByIdAndUserId(commercialValidationId, user.getId())
                .orElseThrow(() -> new RuntimeException(
                        "CommercialValidation not found: " + commercialValidationId));

        log.info("LC cross-validation: swiftAnalysisId={}, commercialValidationId={}, user={}",
                swiftAnalysisId, commercialValidationId, username);

        Map<String, Object> lcPayload         = buildLcPayload(lc);
        Map<String, Object> commercialPayload = buildCommercialPayload(commercial);

        Map<String, Object> requestBody = new LinkedHashMap<>();
        requestBody.put("lc",         lcPayload);
        requestBody.put("commercial", commercialPayload);

        JsonNode result = pipelineClient.callLcCommercialValidate(requestBody);
        return toDTO(lc, commercial, result);
    }

    /**
     * Résultat BRUT de la cohérence LC/MT700 ↔ documents (JSON du pipeline),
     * destiné à l'enrichissement du rapport PDF complet. Réutilise les mêmes
     * builders de payload que {@link #validate}.
     */
    @Transactional(readOnly = true)
    public JsonNode rawCoherence(Long swiftAnalysisId, Long commercialValidationId, String username)
            throws Exception {
        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));
        SwiftAnalysis lc = swiftAnalysisRepo.findById(swiftAnalysisId)
                .orElseThrow(() -> new RuntimeException("SwiftAnalysis not found: " + swiftAnalysisId));
        CommercialValidation commercial = commercialValidationRepo
                .findByIdAndUserId(commercialValidationId, user.getId())
                .orElseThrow(() -> new RuntimeException(
                        "CommercialValidation not found: " + commercialValidationId));

        Map<String, Object> requestBody = new LinkedHashMap<>();
        requestBody.put("lc",         buildLcPayload(lc));
        requestBody.put("commercial", buildCommercialPayload(commercial));
        return pipelineClient.callLcCommercialValidate(requestBody);
    }

    // ── Payload builders ─────────────────────────────────────────────────

    private Map<String, Object> buildLcPayload(SwiftAnalysis lc) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("lc_number",                lc.getLcNumber());
        m.put("applicant",                lc.getApplicant());
        m.put("beneficiary",              lc.getBeneficiary());
        m.put("currency",                 lc.getCurrency());
        m.put("amount",                   lc.getAmount());
        m.put("tolerance_plus",           lc.getTolerancePlus());
        m.put("tolerance_minus",          lc.getToleranceMinus());
        m.put("expiry_date",              lc.getExpiryDate()    != null ? lc.getExpiryDate().toString()    : null);
        m.put("shipment_date",            lc.getShipmentDate()  != null ? lc.getShipmentDate().toString()  : null);
        m.put("presentation_period_days", lc.getPresentationPeriodDays());
        m.put("incoterm",                 lc.getIncoterm());
        m.put("partial_shipments_allowed",lc.getPartialShipmentsAllowed());
        m.put("transhipment_allowed",     lc.getTranshipmentAllowed());
        m.put("goods_description",        lc.getGoodsDescription());
        m.put("documents_required",       lc.getDocumentsRequired());
        m.put("additional_conditions",    lc.getAdditionalConditions());
        return m;
    }

    private Map<String, Object> buildCommercialPayload(CommercialValidation commercial) throws Exception {
        Map<String, Object> result = new LinkedHashMap<>();
        if (commercial.getFullPayload() == null) return result;

        JsonNode root = objectMapper.readTree(commercial.getFullPayload());
        JsonNode docs = root.path("documents");

        List<String> docTypes = new ArrayList<>();
        for (JsonNode doc : docs) {
            String docType = doc.path("doc_type").asText("");
            JsonNode fields = doc.path("fields");
            JsonNode tables = doc.path("tables");

            switch (docType) {
                case "bl" -> {
                    result.put("bl", toMap(fields));
                    result.put("bl_tables", extractFirstTableAsList(tables));
                    docTypes.add("bl");
                }
                case "invoice" -> {
                    result.put("invoice", toMap(fields));
                    result.put("invoice_tables", extractFirstTableAsList(tables));
                    docTypes.add("invoice");
                }
                case "packing_list" -> {
                    result.put("packing_list", toMap(fields));
                    result.put("pl_tables", extractFirstTableAsList(tables));
                    docTypes.add("packing_list");
                }
                default -> log.debug("Unexpected doc_type in commercial payload: {}", docType);
            }
        }
        result.put("_doc_types", docTypes);
        return result;
    }

    private List<Map<String, Object>> extractFirstTableAsList(JsonNode tablesNode) {
        if (tablesNode == null || tablesNode.isMissingNode() || !tablesNode.isObject()) return List.of();
        for (JsonNode arr : tablesNode) {
            if (arr.isArray() && arr.size() > 0) {
                try {
                    return objectMapper.convertValue(arr, new TypeReference<>() {});
                } catch (Exception ignored) {}
            }
        }
        return List.of();
    }

    // ── DTO mapping ───────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private LcCrossValidationResponseDTO toDTO(
            SwiftAnalysis lc,
            CommercialValidation commercial,
            JsonNode result
    ) {
        double score = result.path("final_score").asDouble(0);
        String risk  = result.path("risk_level").asText("unknown").toUpperCase();

        Map<String, BigDecimal> breakdown = new LinkedHashMap<>();
        result.path("score_breakdown").fields().forEachRemaining(e ->
                breakdown.put(e.getKey(), BigDecimal.valueOf(e.getValue().asDouble())));

        Map<String, BigDecimal> weights = new LinkedHashMap<>();
        result.path("weights").fields().forEachRemaining(e ->
                weights.put(e.getKey(), BigDecimal.valueOf(e.getValue().asDouble())));

        List<String> docTypes = new ArrayList<>();
        commercial.getDocuments().forEach(d -> {
            if (d.getDocType() != null) docTypes.add(d.getDocType());
        });

        String decision = result.path("decision").isMissingNode()
                ? null : result.path("decision").asText(null);

        String ucpDecision = result.path("ucp_decision").isMissingNode()
                ? null : result.path("ucp_decision").asText(null);
        Boolean contractualHold = result.path("contractual_hold").isMissingNode()
                ? null : result.path("contractual_hold").asBoolean(false);

        Map<String, BigDecimal> categoryWeights = toBigDecimalMap(result.path("category_weights"));

        return LcCrossValidationResponseDTO.builder()
                .swiftAnalysisId(lc.getId())
                .commercialValidationId(commercial.getId())
                .finalScore(BigDecimal.valueOf(score))
                .riskLevel(risk)
                .decision(decision)
                .ucpDecision(ucpDecision)
                .contractualHold(contractualHold)
                .scoreBreakdown(breakdown)
                .weights(weights)
                .categoryWeights(categoryWeights)
                .categories(toMap(result.path("categories")))
                .crossDocument(toMap(result.path("cross_document")))
                .xai(toMap(result.path("xai")))
                .ruleResults(toListOfMaps(result.path("rule_results")))
                .fuzzyResults(toListOfMaps(result.path("fuzzy_results")))
                .semanticResults(toMap(result.path("semantic_results")))
                .conditionResults(toListOfMaps(result.path("condition_results")))
                .explanations(toListOfMaps(result.path("explanations")))
                .lcSummary(LcCrossValidationResponseDTO.LcSummaryDTO.builder()
                        .lcNumber(lc.getLcNumber())
                        .beneficiary(lc.getBeneficiary())
                        .applicant(lc.getApplicant())
                        .currency(lc.getCurrency())
                        .amount(lc.getAmount())
                        .tolerancePlus(lc.getTolerancePlus())
                        .toleranceMinus(lc.getToleranceMinus())
                        .expiryDate(lc.getExpiryDate()   != null ? lc.getExpiryDate().toString()   : null)
                        .shipmentDate(lc.getShipmentDate() != null ? lc.getShipmentDate().toString() : null)
                        .presentationPeriodDays(lc.getPresentationPeriodDays())
                        .incoterm(lc.getIncoterm())
                        .partialShipmentsAllowed(lc.getPartialShipmentsAllowed())
                        .transhipmentAllowed(lc.getTranshipmentAllowed())
                        .goodsDescription(lc.getGoodsDescription())
                        .additionalConditions(lc.getAdditionalConditions())
                        .build())
                .commercialInfo(LcCrossValidationResponseDTO.CommercialSummaryInfo.builder()
                        .nDocuments(commercial.getNDocuments())
                        .docTypes(docTypes)
                        .riskLevel(commercial.getRiskLevel())
                        .crossScore(commercial.getFinalScore())
                        .build())
                .build();
    }

    // ── JSON helpers ──────────────────────────────────────────────────────

    private List<Map<String, Object>> toListOfMaps(JsonNode node) {
        List<Map<String, Object>> result = new ArrayList<>();
        if (node == null || node.isMissingNode() || !node.isArray()) return result;
        for (JsonNode item : node) {
            try {
                result.add(objectMapper.convertValue(item, new TypeReference<>() {}));
            } catch (Exception ignored) {}
        }
        return result;
    }

    private Map<String, Object> toMap(JsonNode node) {
        if (node == null || node.isMissingNode() || !node.isObject()) return Map.of();
        try { return objectMapper.convertValue(node, new TypeReference<>() {}); }
        catch (Exception e) { return Map.of(); }
    }

    private Map<String, BigDecimal> toBigDecimalMap(JsonNode node) {
        Map<String, BigDecimal> m = new LinkedHashMap<>();
        if (node == null || node.isMissingNode() || !node.isObject()) return m;
        node.fields().forEachRemaining(e ->
                m.put(e.getKey(), BigDecimal.valueOf(e.getValue().asDouble())));
        return m;
    }
}
