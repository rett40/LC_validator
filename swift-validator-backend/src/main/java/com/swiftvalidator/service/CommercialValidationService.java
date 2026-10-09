package com.swiftvalidator.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.swiftvalidator.dto.CommercialValidationResponseDTO;
import com.swiftvalidator.entity.*;
import com.swiftvalidator.repository.AuditLogRepository;
import com.swiftvalidator.repository.CommercialValidationRepository;
import com.swiftvalidator.repository.DocumentRepository;
import com.swiftvalidator.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.math.BigDecimal;
import java.util.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class CommercialValidationService {

    private final PipelineClientService pipelineClient;
    private final DocumentStorageService documentStorage;
    private final CommercialValidationRepository repo;
    private final UserRepository userRepo;
    private final DocumentRepository documentRepo;
    private final AuditLogRepository auditLogRepo;
    private final ObjectMapper objectMapper;

    @Transactional
    public CommercialValidationResponseDTO validate(
            List<MultipartFile> files,
            String username,
            String lcReference,
            Long swiftAnalysisId
    ) throws Exception {
        long t0 = System.currentTimeMillis();

        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        log.info("Commercial validation: {} file(s) for user={}, lcRef={}, swiftAnalysisId={}",
                files.size(), username, lcReference, swiftAnalysisId);

        JsonNode pipelineResult = pipelineClient.analyzeCommercialDocuments(files);
        log.info("Commercial pipeline responded, building entity...");

        CommercialValidation validation = buildValidation(pipelineResult, user, files, lcReference, swiftAnalysisId);
        validation.setProcessingTimeS(
                BigDecimal.valueOf((System.currentTimeMillis() - t0) / 1000.0)
        );

        // Store each file and persist the Document before attaching to CommercialDocument
        for (int i = 0; i < files.size(); i++) {
            MultipartFile file = files.get(i);
            try {
                Document doc = documentStorage.store(file, user);
                Document savedDoc = documentRepo.save(doc);
                if (i < validation.getDocuments().size()) {
                    validation.getDocuments().get(i).setDocument(savedDoc);
                }
            } catch (Exception e) {
                log.warn("Could not store file {}: {}", file.getOriginalFilename(), e.getMessage());
            }
        }

        repo.save(validation);
        log.info("Commercial validation saved with id={}", validation.getId());

        auditLogRepo.save(AuditLog.builder()
                .action("COMMERCIAL_VALIDATION_CREATED")
                .entityType("CommercialValidation")
                .entityId(validation.getId())
                .userId(user.getId())
                .username(user.getUsername())
                .details("LC: " + lcReference + " | Docs: " + files.size()
                        + " | Risk: " + validation.getRiskLevel())
                .build());

        return toDTO(validation, true);
    }

    @Transactional(readOnly = true)
    public CommercialValidationResponseDTO getById(Long id, String username) {
        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return repo.findByIdAndUserId(id, user.getId())
                .map(v -> toDTO(v, true))
                .orElseThrow(() -> new RuntimeException("Validation not found: " + id));
    }

    @Transactional(readOnly = true)
    public Page<CommercialValidationResponseDTO> list(String username, Pageable pageable) {
        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return repo.findByUserIdOrderByCreatedAtDesc(user.getId(), pageable)
                .map(v -> toDTO(v, false));
    }

    @Transactional(readOnly = true)
    public Page<CommercialValidationResponseDTO> listByAnalysis(
            String username, Long swiftAnalysisId, Pageable pageable) {
        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return repo.findBySwiftAnalysisIdAndUserIdOrderByCreatedAtDesc(
                        swiftAnalysisId, user.getId(), pageable)
                .map(v -> toDTO(v, false));
    }

    @Transactional(readOnly = true)
    public Page<CommercialValidationResponseDTO> search(
            String username, String riskLevel, String lcReference, Pageable pageable
    ) {
        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return repo.search(user.getId(), riskLevel, lcReference, pageable)
                .map(v -> toDTO(v, false));
    }

    // ── Build entity from pipeline JSON ──────────────────────────────────────

    private CommercialValidation buildValidation(
            JsonNode root, User user, List<MultipartFile> files, String lcReference, Long swiftAnalysisId
    ) {
        JsonNode summary = root.path("summary");
        JsonNode crossReport = root.path("cross_report");

        String riskLevel = crossReport.path("risk_level").asText(null);
        double finalScore = crossReport.path("final_score").asDouble(-1);

        CommercialValidation validation = CommercialValidation.builder()
                .lcReference(lcReference)
                .swiftAnalysisId(swiftAnalysisId)
                .user(user)
                .status("COMPLETED")
                .riskLevel(riskLevel != null ? riskLevel.toUpperCase() : null)
                .finalScore(finalScore >= 0 ? BigDecimal.valueOf(finalScore) : null)
                .nDocuments(summary.path("total_documents").asInt(files.size()))
                .nProcessed(summary.path("processed").asInt(0))
                .nFailed(summary.path("failed").asInt(0))
                .crossReport(crossReport.toString())
                .fullPayload(root.toString())
                .build();

        List<CommercialDocument> docs = new ArrayList<>();
        JsonNode docsNode = root.path("documents");
        int fileIdx = 0;
        for (JsonNode docNode : docsNode) {
            String docStatus = docNode.path("status").asText("error");
            String sourceFilename = docNode.path("source").asText(
                    fileIdx < files.size() ? files.get(fileIdx).getOriginalFilename() : "unknown"
            );

            CommercialDocument.CommercialDocumentBuilder builder = CommercialDocument.builder()
                    .validation(validation)
                    .sourceFilename(sourceFilename)
                    .status(docStatus);

            if ("success".equals(docStatus)) {
                JsonNode ruleSummary = docNode.path("rule_summary");
                JsonNode scoreNode = docNode.path("extraction_score");
                double extractionScore = scoreNode.path("global_score").asDouble(
                        scoreNode.asDouble(-1)
                );

                builder
                        .docType(docNode.path("doc_type").asText(null))
                        .inputType(docNode.path("input_type").asText(null))
                        .confidence(bigDecimal(docNode, "confidence"))
                        .extractionScore(extractionScore >= 0 ? BigDecimal.valueOf(extractionScore) : null)
                        .nFields(docNode.path("fields").size())
                        .nPages(docNode.path("n_pages").asInt(0))
                        .nWarnings(docNode.path("warnings").size())
                        .nRuleFails(ruleSummary.path("fail").asInt(0))
                        .fieldsJson(docNode.path("fields").toString())
                        .tablesJson(resolveTablesJson(docNode))
                        .tiersJson(docNode.path("extraction_tiers").toString())
                        .ruleResultsJson(docNode.path("rule_results").toString())
                        .processingTimeS(bigDecimal(docNode, "processing_time_s"));
            }

            docs.add(builder.build());
            fileIdx++;
        }
        validation.setDocuments(docs);
        return validation;
    }

    // ── DTO mapping ───────────────────────────────────────────────────────────

    public CommercialValidationResponseDTO toDTO(CommercialValidation v, boolean includePayload) {
        // Parse fullPayload once to use as authoritative source for document tables.
        // This avoids Hibernate JSONB→String round-trip issues with @JdbcTypeCode(SqlTypes.JSON).
        JsonNode payloadDocs = null;
        if (v.getFullPayload() != null) {
            try {
                payloadDocs = objectMapper.readTree(v.getFullPayload()).path("documents");
                if (payloadDocs.isMissingNode()) payloadDocs = null;
            } catch (Exception e) {
                log.debug("Could not parse fullPayload for table extraction: {}", e.getMessage());
            }
        }

        List<CommercialValidationResponseDTO.DocumentResultDTO> docDTOs = new ArrayList<>();
        List<CommercialDocument> docs = v.getDocuments();
        for (int i = 0; i < docs.size(); i++) {
            JsonNode payloadDoc = (payloadDocs != null && i < payloadDocs.size())
                    ? payloadDocs.get(i) : null;
            docDTOs.add(toDocumentDTO(docs.get(i), payloadDoc));
        }

        CommercialValidationResponseDTO.CrossReportDTO crossReportDTO = null;
        List<Map<String, Object>> crossValidationList = null;

        if (v.getCrossReport() != null) {
            try {
                JsonNode cr = objectMapper.readTree(v.getCrossReport());
                crossReportDTO = buildCrossReportDTO(cr);
                crossValidationList = toListOfMaps(cr.path("rule_results"));
            } catch (Exception e) {
                log.debug("Could not parse cross_report: {}", e.getMessage());
            }
        }

        return CommercialValidationResponseDTO.builder()
                .id(v.getId())
                .lcReference(v.getLcReference())
                .swiftAnalysisId(v.getSwiftAnalysisId())
                .status(v.getStatus())
                .riskLevel(v.getRiskLevel())
                .finalScore(v.getFinalScore())
                .nDocuments(v.getNDocuments())
                .nProcessed(v.getNProcessed())
                .nFailed(v.getNFailed())
                .processingTimeS(v.getProcessingTimeS())
                .createdAt(v.getCreatedAt())
                .summary(CommercialValidationResponseDTO.SummaryDTO.builder()
                        .totalDocuments(v.getNDocuments())
                        .processed(v.getNProcessed())
                        .failed(v.getNFailed())
                        .processingTimeS(v.getProcessingTimeS())
                        .build())
                .documents(docDTOs)
                .crossValidation(crossValidationList)
                .crossReport(crossReportDTO)
                .fullPayloadJson(includePayload ? v.getFullPayload() : null)
                .build();
    }

    private CommercialValidationResponseDTO.DocumentResultDTO toDocumentDTO(
            CommercialDocument doc, JsonNode payloadDoc) {
        Map<String, Object> fields = jsonToMap(doc.getFieldsJson());
        Map<String, String> tiers = jsonToStringMap(doc.getTiersJson());

        // Tables: prefer the raw pipeline JSON node to avoid Hibernate JSONB round-trip issues.
        Map<String, Object> tables;
        if (payloadDoc != null) {
            tables = resolveTablesFromNode(payloadDoc);
        } else {
            tables = jsonToMap(doc.getTablesJson());
        }
        if (tables.isEmpty()) {
            log.debug("Tables empty for {} — tablesJson={}", doc.getSourceFilename(),
                    doc.getTablesJson() != null ? doc.getTablesJson().substring(0, Math.min(80, doc.getTablesJson().length())) : "null");
        }
        List<Map<String, Object>> ruleResults = (payloadDoc != null)
                ? toListOfMaps(payloadDoc.path("rule_results"))
                : toListOfMaps(parseNode(doc.getRuleResultsJson()));

        int nRuleFails = ruleResults.stream()
                .mapToInt(r -> "fail".equals(r.get("status")) ? 1 : 0)
                .sum();
        int nRulePass = ruleResults.stream()
                .mapToInt(r -> "pass".equals(r.get("status")) ? 1 : 0)
                .sum();
        int nRuleWarn = ruleResults.stream()
                .mapToInt(r -> "warning".equals(r.get("status")) ? 1 : 0)
                .sum();

        return CommercialValidationResponseDTO.DocumentResultDTO.builder()
                .id(doc.getId())
                .sourceFilename(doc.getSourceFilename())
                .status(doc.getStatus())
                .docType(doc.getDocType())
                .inputType(doc.getInputType())
                .confidence(doc.getConfidence())
                .confident(doc.getConfidence() != null && doc.getConfidence().doubleValue() >= 0.80)
                .extractionScore(doc.getExtractionScore())
                .nFields(doc.getNFields())
                .nPages(doc.getNPages())
                .nWarnings(doc.getNWarnings())
                .nRuleFails(nRuleFails)
                .fields(fields)
                .tables(tables)
                .extractionTiers(tiers)
                .ruleResults(ruleResults)
                .ruleSummary(CommercialValidationResponseDTO.RuleSummaryDTO.builder()
                        .pass(nRulePass)
                        .fail(nRuleFails)
                        .warning(nRuleWarn)
                        .corrected(0)
                        .build())
                .processingTimeS(doc.getProcessingTimeS())
                .build();
    }

    @SuppressWarnings("unchecked")
    private CommercialValidationResponseDTO.CrossReportDTO buildCrossReportDTO(JsonNode cr) {
        double score = cr.path("final_score").asDouble(-1);
        String risk = cr.path("risk_level").asText(null);

        Map<String, BigDecimal> breakdown = new LinkedHashMap<>();
        JsonNode bd = cr.path("score_breakdown");
        if (!bd.isMissingNode()) {
            bd.fields().forEachRemaining(e -> {
                try { breakdown.put(e.getKey(), BigDecimal.valueOf(e.getValue().asDouble())); }
                catch (Exception ignored) {}
            });
        }

        Map<String, BigDecimal> semanticScores = new LinkedHashMap<>();
        JsonNode ss = cr.path("semantic_scores");
        if (!ss.isMissingNode()) {
            ss.fields().forEachRemaining(e -> {
                try { semanticScores.put(e.getKey(), BigDecimal.valueOf(e.getValue().asDouble())); }
                catch (Exception ignored) {}
            });
        }

        return CommercialValidationResponseDTO.CrossReportDTO.builder()
                .finalScore(score >= 0 ? BigDecimal.valueOf(score) : null)
                .riskLevel(risk != null ? risk.toUpperCase() : null)
                .scoreBreakdown(breakdown)
                .ruleResults(toListOfMaps(cr.path("rule_results")))
                .semanticScores(semanticScores)
                .nliResults(toListOfMaps(cr.path("nli_results")))
                .tableMatches(toListOfMaps(cr.path("table_matches")))
                .explanations(toListOfMaps(cr.path("explanations")))
                .documents(toListOfMaps(cr.path("documents")))
                .build();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Mirrors Streamlit's two-tier table display: prefer named tables from the pipeline;
     * fall back to structured_rows (wrapped in a doc-type-specific key) when tables is empty.
     */
    private String resolveTablesJson(JsonNode docNode) {
        JsonNode tables = docNode.path("tables");
        if (!tables.isMissingNode() && tables.isObject() && tables.size() > 0) {
            // Only treat tables as valid if at least one key has a non-empty array value.
            // Otherwise fall through to structured_rows to mirror Streamlit's fallback logic.
            boolean hasRows = false;
            for (JsonNode arr : tables) {
                if (arr.isArray() && arr.size() > 0) { hasRows = true; break; }
            }
            if (hasRows) return tables.toString();
        }
        JsonNode structuredRows = docNode.path("structured_rows");
        if (structuredRows.isArray() && structuredRows.size() > 0) {
            String docType = docNode.path("doc_type").asText("");
            String tableName = switch (docType) {
                case "bl"           -> "containers";
                case "invoice"      -> "invoice_items";
                case "packing_list" -> "packages";
                default             -> "rows";
            };
            com.fasterxml.jackson.databind.node.ObjectNode result = objectMapper.createObjectNode();
            result.set(tableName, structuredRows);
            return result.toString();
        }
        return tables.isMissingNode() ? "{}" : tables.toString();
    }

    /**
     * Resolves the tables map directly from a pipeline document JsonNode.
     * Prefers named tables; falls back to structured_rows wrapped in a doc-type key.
     * Uses toListOfMaps() explicitly so each table value is List<Map<String,Object>>,
     * which Spring serializes as a JSON array that Angular's Array.isArray() accepts.
     */
    private Map<String, Object> resolveTablesFromNode(JsonNode docNode) {
        JsonNode tablesNode = docNode.path("tables");
        if (!tablesNode.isMissingNode() && tablesNode.isObject() && tablesNode.size() > 0) {
            Map<String, Object> result = new LinkedHashMap<>();
            tablesNode.fields().forEachRemaining(entry -> {
                JsonNode arrNode = entry.getValue();
                if (arrNode.isArray() && arrNode.size() > 0) {
                    List<Map<String, Object>> rows = toListOfMaps(arrNode);
                    if (!rows.isEmpty()) result.put(entry.getKey(), rows);
                }
            });
            if (!result.isEmpty()) return result;
        }
        JsonNode structuredRows = docNode.path("structured_rows");
        if (structuredRows.isArray() && structuredRows.size() > 0) {
            String docType = docNode.path("doc_type").asText("");
            String tableName = switch (docType) {
                case "bl"           -> "containers";
                case "invoice"      -> "invoice_items";
                case "packing_list" -> "packages";
                default             -> "rows";
            };
            List<Map<String, Object>> rows = toListOfMaps(structuredRows);
            Map<String, Object> result = new LinkedHashMap<>();
            result.put(tableName, rows);
            return result;
        }
        return Map.of();
    }

    private BigDecimal bigDecimal(JsonNode node, String field) {
        JsonNode val = node.path(field);
        if (val.isMissingNode() || val.isNull()) return null;
        try { return BigDecimal.valueOf(val.asDouble()); } catch (Exception e) { return null; }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> jsonToMap(String json) {
        if (json == null || json.isBlank() || "null".equals(json)) return Map.of();
        try {
            return objectMapper.readValue(json, new TypeReference<>() {});
        } catch (Exception e) { return Map.of(); }
    }

    @SuppressWarnings("unchecked")
    private Map<String, String> jsonToStringMap(String json) {
        if (json == null || json.isBlank() || "null".equals(json)) return Map.of();
        try {
            return objectMapper.readValue(json, new TypeReference<>() {});
        } catch (Exception e) { return Map.of(); }
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> toListOfMaps(JsonNode node) {
        List<Map<String, Object>> result = new ArrayList<>();
        if (node == null || node.isMissingNode() || !node.isArray()) return result;
        for (JsonNode item : node) {
            try {
                Map<String, Object> map = objectMapper.convertValue(item, new TypeReference<>() {});
                result.add(map);
            } catch (Exception ignored) {}
        }
        return result;
    }

    private JsonNode parseNode(String json) {
        if (json == null || json.isBlank()) return objectMapper.createArrayNode();
        try { return objectMapper.readTree(json); }
        catch (Exception e) { return objectMapper.createArrayNode(); }
    }
}
