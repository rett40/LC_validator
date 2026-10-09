package com.swiftvalidator.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.swiftvalidator.dto.AnalysisResponseDTO;
import com.swiftvalidator.dto.DashboardStatsDTO;
import com.swiftvalidator.entity.*;
import com.swiftvalidator.mapper.AnalysisMapper;
import com.swiftvalidator.repository.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class AnalysisService {

    private final PipelineClientService pipelineClient;
    private final DocumentStorageService documentStorage;
    private final SwiftAnalysisRepository analysisRepo;
    private final UcpFindingRepository ucpFindingRepo;
    private final NlpClauseRepository nlpClauseRepo;
    private final UserRepository userRepo;
    private final AuditLogRepository auditLogRepo;
    private final AnalysisMapper mapper;
    private final ObjectMapper objectMapper;

    @Transactional
    public AnalysisResponseDTO analyzeFile(MultipartFile file, String username) throws Exception {
        long globalStart = System.currentTimeMillis();

        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        Document doc = documentStorage.store(file, user);

        log.info("Sending file to Python pipeline: {}", file.getOriginalFilename());
        JsonNode pipelineResult = pipelineClient.analyze(file);
        log.info("Pipeline payload received, building analysis entity...");

        SwiftAnalysis analysis = buildAnalysis(pipelineResult, user, doc);
        analysis.setProcessingMs(System.currentTimeMillis() - globalStart);
        log.info("Analysis entity built (swiftType={}, riskLevel={}), saving to DB...", analysis.getSwiftType(), analysis.getRiskLevel());

        analysisRepo.save(analysis);
        log.info("Analysis saved successfully with id={}", analysis.getId());

        auditLogRepo.save(AuditLog.builder()
                .action("ANALYSIS_CREATED")
                .entityType("SwiftAnalysis")
                .entityId(analysis.getId())
                .userId(user.getId())
                .username(user.getUsername())
                .details("LC: " + analysis.getLcNumber() + " | Risk: " + analysis.getRiskLevel())
                .build());
        log.info("Audit log saved, mapping response DTO...");

        AnalysisResponseDTO dto = mapper.toDTO(analysis, true);
        log.info("Analysis DTO mapped, returning response for id={}", dto.getId());
        return dto;
    }

    @Transactional(readOnly = true)
    public AnalysisResponseDTO getById(Long id, String username) {
        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return analysisRepo.findByIdAndUserId(id, user.getId())
                .map(analysis -> mapper.toDTO(analysis, true))
                .orElseThrow(() -> new RuntimeException("Analysis not found: " + id));
    }

    @Transactional(readOnly = true)
    public Page<AnalysisResponseDTO> listByUser(String username, Pageable pageable) {
        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return analysisRepo.findByUserIdOrderByCreatedAtDesc(user.getId(), pageable)
                .map(analysis -> mapper.toDTO(analysis, false));
    }

    @Transactional(readOnly = true)
    public Page<AnalysisResponseDTO> search(String username, String riskLevel, String lcNumber, Pageable pageable) {
        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return analysisRepo.search(user.getId(), riskLevel, lcNumber, pageable)
                .map(analysis -> mapper.toDTO(analysis, false));
    }

    @Transactional(readOnly = true)
    public DashboardStatsDTO getDashboardStats(String username) {
        User user = userRepo.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));

        Long uid = user.getId();

        return DashboardStatsDTO.builder()
                .totalAnalyses(analysisRepo.countByUserId(uid))
                .criticalCount(analysisRepo.countByUserIdAndRiskLevel(uid, "CRITICAL"))
                .highCount(analysisRepo.countByUserIdAndRiskLevel(uid, "HIGH"))
                .mediumCount(analysisRepo.countByUserIdAndRiskLevel(uid, "MEDIUM"))
                .cleanCount(analysisRepo.countByUserIdAndRiskLevel(uid, "CLEAN"))
                .last24hCount(analysisRepo.countSince(LocalDateTime.now().minusHours(24)))
                .avgUcpScore(analysisRepo.avgUcpScore(uid))
                .build();
    }

    private SwiftAnalysis buildAnalysis(JsonNode root, User user, Document doc) {
        JsonNode swift = root.path("swift").path("entities");
        JsonNode ucp = root.path("ucp_compliance");
        JsonNode ds = root.path("ds_analysis");
        JsonNode ext = root.path("external_validation");
        JsonNode nlp = root.path("swift").path("nlp");
        JsonNode ocrNode = root.path("ocr");

        SwiftAnalysis analysis = SwiftAnalysis.builder()
                .lcNumber(limit(text(swift, "lc_number"), 50))
                .swiftType(limit(text(root.path("swift"), "swift_type"), 10))
                .lcForm(limit(text(swift, "lc_form"), 30))
                .applicant(limit(text(swift, "applicant"), 500))
                .beneficiary(limit(text(swift, "beneficiary"), 500))
                .issuingBank(limit(text(swift, "issuing_bank"), 200))
                .currency(limit(text(swift, "currency"), 3))
                .amount(decimal(swift, "amount"))
                .tolerancePlus(intVal(swift, "tolerance_plus"))
                .toleranceMinus(intVal(swift, "tolerance_minus"))
                .issueDate(date(swift, "issue_date"))
                .expiryDate(date(swift, "expiry_date"))
                .expiryPlace(limit(text(swift, "expiry_place"), 100))
                .shipmentDate(date(swift, "shipment_date"))
                .presentationPeriodDays(intVal(swift, "presentation_period"))
                .incoterm(limit(text(swift, "incoterm"), 10))
                .country(limit(text(swift, "country"), 100))
                .partialShipmentsAllowed(bool(swift, "partial_shipments"))
                .transhipmentAllowed(bool(swift, "transhipment"))
                .confirmationInstructions(limit(text(swift, "confirmation"), 50))
                .applicableRules(limit(text(swift, "applicable_rules"), 100))
                .goodsDescription(text(swift, "goods_description"))
                .documentsRequired(text(swift, "documents_required"))
                .additionalConditions(text(swift, "additional_conditions"))
                .chargesDetails(text(swift, "charges"))
                .ucpComplianceScore(intVal(ucp, "global_score"))
                .nlpScore(intVal(nlp, "score"))
                .completenessScore(intVal(ds.path("features"), "completeness_score"))
                .semanticScore(intVal(ds.path("features"), "semantic_score"))
                .riskLevel(limit(text(ds.path("risk"), "risk_level"), 20))
                .riskScore(decimal(ds.path("risk"), "risk_score"))
                .mlRiskScore(decimal(ds.path("features"), "ml_risk_score"))
                .paymentBlocked(ucp.path("payment_blocked").asBoolean(false))
                .countryRisk(limit(text(ext, "risk_level"), 20))
                .nUcpErrors(intVal(ucp.path("summary"), "errors"))
                .nUcpWarnings(intVal(ucp.path("summary"), "warnings"))
                .nUcpExclusions(intVal(ucp.path("summary"), "exclusions"))
                .nClauses(intVal(ds.path("features"), "n_clauses"))
                .nCriticalClauses(intVal(ds.path("features"), "n_critical_clauses"))
                .nHighRiskClauses(intVal(ds.path("features"), "n_high_risk_clauses"))
                .rawText(text(ocrNode, "text"))
                .ocrConfidence(decimal(ocrNode, "confidence"))
                .fullPayload(root.toString())
                .status(limit("COMPLETED", 20))
                .user(user)
                .document(doc)
                .build();

        List<UcpFinding> findingList = new ArrayList<>();
        for (JsonNode f : ucp.path("findings")) {
            findingList.add(UcpFinding.builder()
                    .analysis(analysis)
                    .article(text(f, "article"))
                    .articleTitle(text(f, "article_title"))
                    .code(text(f, "code"))
                    .category(text(f, "category"))
                    .severity(text(f, "severity"))
                    .tier(intVal(f, "tier"))
                    .weight(decimal(f, "weight"))
                    .message(text(f, "message"))
                    .clauseText(text(f, "clause_text"))
                    .recommendation(text(f, "recommendation"))
                    .blocksPayment(f.path("blocks_payment").asBoolean(false))
                    .cascadeImpact(intVal(f, "cascade_impact"))
                    .exclusionType(text(f, "exclusion_type"))
                    .confidence(decimal(f, "confidence"))
                    .findingNature(text(f, "finding_nature"))
                    .impactLevel(text(f, "impact_level"))
                    .clauseType(text(f, "clause_type"))
                    .blocksPaymentReason(text(f, "blocks_payment_reason"))
                    .build());
        }
        analysis.setUcpFindings(findingList);

        List<NlpClause> clauseList = new ArrayList<>();
        for (JsonNode c : nlp.path("clauses")) {
            clauseList.add(NlpClause.builder()
                    .analysis(analysis)
                    .text(text(c, "text"))
                    .clauseType(text(c, "type"))
                    .importance(text(c, "importance"))
                    .riskLevel(text(c, "risk_level"))
                    .confidence(decimal(c, "confidence"))
                    .method(text(c, "method"))
                    .ucpRefs(c.path("ucp_refs").toString())
                    .build());
        }
        analysis.setNlpClauses(clauseList);

        ExternalValidation externalValidation = ExternalValidation.builder()
                .analysis(analysis)
                .riskLevel(limit(text(ext, "risk_level"), 20))
                .rawScore(decimal(ext, "risk_score"))
                .approach(limit(text(ext, "approach"), 10))
                .currencyCode(limit(text(ext.path("details").path("currency"), "code"), 3))
                .currencyValid(ext.path("details").path("currency").path("is_valid").asBoolean(true))
                .detailsJson(ext.path("details").toString())
                .processingMs(ext.path("processing_ms").asLong(0))
                .processedAt(LocalDateTime.now())
                .build();
        analysis.setExternalValidation(externalValidation);

        DsRiskScore dsRiskScore = DsRiskScore.builder()
                .analysis(analysis)
                .riskLevel(text(ds.path("risk"), "risk_level"))
                .riskScore(decimal(ds.path("risk"), "risk_score"))
                .mlRiskScore(decimal(ds.path("features"), "ml_risk_score"))
                .modelUsed(text(ds.path("risk"), "model_used"))
                .completenessScore(intVal(ds.path("features"), "completeness_score"))
                .nClauses(intVal(ds.path("features"), "n_clauses"))
                .nCriticalClauses(intVal(ds.path("features"), "n_critical_clauses"))
                .incotermRisk(intVal(ds.path("features"), "incoterm_risk"))
                .presentationDays(intVal(ds.path("features"), "presentation_days"))
                .validityDays(intVal(ds.path("features"), "validity_days"))
                .processingMs(ds.path("processing_ms").asLong(0))
                .build();
        analysis.setDsRiskScore(dsRiskScore);

        return analysis;
    }

    private String text(JsonNode node, String field) {
        JsonNode val = node.path(field);
        return val.isNull() || val.isMissingNode() ? null : val.asText();
    }

    private String limit(String value, int maxLen) {
        if (value == null) return null;
        return value.length() <= maxLen ? value : value.substring(0, maxLen);
    }

    private Integer intVal(JsonNode node, String field) {
        JsonNode val = node.path(field);
        return val.isNull() || val.isMissingNode() ? null : val.asInt();
    }

    private BigDecimal decimal(JsonNode node, String field) {
        JsonNode val = node.path(field);
        if (val.isNull() || val.isMissingNode()) return null;
        try {
            return new BigDecimal(val.asText());
        } catch (Exception e) {
            return null;
        }
    }

    private Boolean bool(JsonNode node, String field) {
        JsonNode val = node.path(field);
        if (val.isNull() || val.isMissingNode()) return null;
        if (val.isBoolean()) return val.asBoolean();
        String s = val.asText().toUpperCase();
        return "ALLOWED".equals(s) || "TRUE".equals(s) || "YES".equals(s);
    }

    private LocalDate date(JsonNode node, String field) {
        String v = text(node, field);
        if (v == null) return null;
        try {
            return LocalDate.parse(v);
        } catch (Exception e) {
            return null;
        }
    }
}
