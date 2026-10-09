package com.swiftvalidator.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.swiftvalidator.dto.ReportDTO;
import com.swiftvalidator.entity.AnalysisReport;
import com.swiftvalidator.entity.CommercialValidation;
import com.swiftvalidator.entity.SwiftAnalysis;
import com.swiftvalidator.repository.AnalysisReportRepository;
import com.swiftvalidator.repository.CommercialValidationRepository;
import com.swiftvalidator.repository.SwiftAnalysisRepository;
import com.swiftvalidator.repository.UserRepository;
import io.minio.GetPresignedObjectUrlArgs;
import io.minio.MinioClient;
import io.minio.http.Method;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class ReportService {

    private final AnalysisReportRepository       reportRepo;
    private final SwiftAnalysisRepository        analysisRepo;
    private final UserRepository                 userRepo;
    private final MinioClient                    presignClient;   // endpoint PUBLIC (URLs navigateur)
    private final RestTemplate                   restTemplate;
    private final ObjectMapper                   objectMapper;
    private final CommercialValidationRepository commercialValidationRepo;
    private final LcCrossValidationService       lcCrossService;

    public ReportService(AnalysisReportRepository reportRepo,
                         SwiftAnalysisRepository analysisRepo,
                         UserRepository userRepo,
                         @Qualifier("presignMinioClient") MinioClient presignClient,
                         RestTemplate restTemplate,
                         ObjectMapper objectMapper,
                         CommercialValidationRepository commercialValidationRepo,
                         LcCrossValidationService lcCrossService) {
        this.reportRepo               = reportRepo;
        this.analysisRepo             = analysisRepo;
        this.userRepo                 = userRepo;
        this.presignClient            = presignClient;
        this.restTemplate             = restTemplate;
        this.objectMapper             = objectMapper;
        this.commercialValidationRepo = commercialValidationRepo;
        this.lcCrossService           = lcCrossService;
    }

    /** Résout l'id utilisateur depuis le username (null si introuvable — colonne nullable). */
    private Long resolveUserId(String username) {
        if (username == null) return null;
        return userRepo.findByUsername(username).map(u -> u.getId()).orElse(null);
    }

    @Value("${minio.bucket-reports:swift-reports}")
    private String reportsBucket;

    @Value("${pipeline.base-url}")
    private String pythonBaseUrl;

    // ── Générer rapport MT700 ────────────────────────────────────────────
    public ReportDTO.ReportResponse generateMt700Report(Long analysisId, String username) {
        Long userId = resolveUserId(username);
        SwiftAnalysis analysis = analysisRepo.findById(analysisId)
            .orElseThrow(() -> new IllegalArgumentException("Analyse introuvable : " + analysisId));

        Map<String, Object> payload = buildMt700Payload(analysis);
        Map<?, ?> result = callPythonPdf("/pdf/mt700/" + analysisId, payload);

        AnalysisReport report = saveReport(
            analysisId, "MT700_REPORT",
            "Rapport MT700 — " + analysis.getLcNumber(),
            (String) result.get("object_name"),
            toLong(result.get("file_size")),
            userId
        );

        // Marquer l'analyse comme MT700_ONLY si pas déjà full
        if (!"FULL_VALIDATION".equals(analysis.getValidationType())) {
            analysis.setValidationType("MT700_ONLY");
            analysisRepo.save(analysis);
        }

        return toResponse(report);
    }

    // ── Générer rapport complet ──────────────────────────────────────────
    public ReportDTO.ReportResponse generateFullReport(Long analysisId,
                                                       Long commercialValidationId,
                                                       String username) {
        Long userId = resolveUserId(username);
        SwiftAnalysis analysis = analysisRepo.findById(analysisId)
            .orElseThrow(() -> new IllegalArgumentException("Analyse introuvable : " + analysisId));

        Map<String, Object> payload = buildFullPayload(analysis, commercialValidationId, username);
        Map<?, ?> result = callPythonPdf("/pdf/full/" + analysisId, payload);

        AnalysisReport report = saveReport(
            analysisId, "FULL_REPORT",
            "Rapport Complet — " + analysis.getLcNumber(),
            (String) result.get("object_name"),
            toLong(result.get("file_size")),
            userId
        );

        analysis.setValidationType("FULL_VALIDATION");
        analysisRepo.save(analysis);

        return toResponse(report);
    }

    // ── Lister les rapports d'une analyse ───────────────────────────────
    public List<ReportDTO.ReportResponse> listReports(Long analysisId) {
        return reportRepo.findByAnalysisIdOrderByGeneratedAtDesc(analysisId)
            .stream().map(this::toResponse).toList();
    }

    // ── URL présignée fraîche (7 jours) ─────────────────────────────────
    public String freshDownloadUrl(Long reportId) {
        AnalysisReport report = reportRepo.findById(reportId)
            .orElseThrow(() -> new IllegalArgumentException("Rapport introuvable : " + reportId));
        return presignedUrl(report.getMinioObjectKey());
    }

    // ── Helpers privés ───────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private Map<?, ?> callPythonPdf(String path, Map<String, Object> payload) {
        String url = pythonBaseUrl + path;
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> req = new HttpEntity<>(payload, headers);
        ResponseEntity<Map> resp = restTemplate.exchange(url, HttpMethod.POST, req, Map.class);
        if (!resp.getStatusCode().is2xxSuccessful() || resp.getBody() == null)
            throw new RuntimeException("Erreur génération PDF Python : " + resp.getStatusCode());
        return resp.getBody();
    }

    private AnalysisReport saveReport(Long analysisId, String type, String displayName,
                                       String objectKey, Long fileSize, Long userId) {
        AnalysisReport r = AnalysisReport.builder()
            .analysisId(analysisId)
            .reportType(type)
            .displayName(displayName)
            .minioObjectKey(objectKey)
            .fileSize(fileSize)
            .generatedAt(LocalDateTime.now())
            .generatedBy(userId)
            .build();
        return reportRepo.save(r);
    }

    private String presignedUrl(String objectKey) {
        try {
            return presignClient.getPresignedObjectUrl(
                GetPresignedObjectUrlArgs.builder()
                    .bucket(reportsBucket)
                    .object(objectKey)
                    .method(Method.GET)
                    .expiry(7, TimeUnit.DAYS)
                    .build()
            );
        } catch (Exception e) {
            log.error("Erreur génération URL présignée : {}", e.getMessage());
            throw new RuntimeException("Impossible de générer l'URL de téléchargement", e);
        }
    }

    private ReportDTO.ReportResponse toResponse(AnalysisReport r) {
        return ReportDTO.ReportResponse.builder()
            .id(r.getId())
            .analysisId(r.getAnalysisId())
            .reportType(r.getReportType())
            .displayName(r.getDisplayName())
            .fileSize(r.getFileSize())
            .generatedAt(r.getGeneratedAt())
            .downloadUrl(presignedUrl(r.getMinioObjectKey()))
            .build();
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> buildMt700Payload(SwiftAnalysis a) {
        Map<String, Object> p = new java.util.HashMap<>();
        p.put("lc_number",    a.getLcNumber());
        p.put("applicant",    a.getApplicant());
        p.put("beneficiary",  a.getBeneficiary());
        p.put("issuing_bank", a.getIssuingBank());
        p.put("currency",     a.getCurrency());
        p.put("amount",       a.getAmount());
        p.put("issue_date",   a.getIssueDate() != null ? a.getIssueDate().toString() : null);
        p.put("expiry_date",  a.getExpiryDate() != null ? a.getExpiryDate().toString() : null);
        p.put("incoterm",     a.getIncoterm());
        p.put("country",      a.getCountry());
        p.put("validity_days", a.getValidityDays());
        p.put("presentation_period_days", a.getPresentationPeriodDays());
        p.put("goods_description",  a.getGoodsDescription());
        p.put("documents_required", a.getDocumentsRequired());
        p.put("risk_level",    a.getRiskLevel());
        p.put("risk_score",    a.getRiskScore());
        p.put("ml_risk_score", a.getMlRiskScore());
        p.put("ucp_compliance_score", a.getUcpComplianceScore());
        p.put("nlp_score",     a.getNlpScore());
        p.put("completeness_score", a.getCompletenessScore());
        p.put("payment_blocked", a.getPaymentBlocked());
        p.put("n_ucp_errors",   a.getNUcpErrors());
        p.put("n_ucp_warnings", a.getNUcpWarnings());
        // Full payload JSON (contient risk_ml, explanation, ucp_findings…)
        if (a.getFullPayload() != null) {
            try {
                Map<?, ?> full = objectMapper.readValue(a.getFullPayload(), Map.class);
                p.put("risk_ml", full.get("risk_ml"));
                p.put("external_validation", full.get("external_validation"));
                p.put("ucp_findings", full.get("ucp_findings"));
            } catch (Exception e) {
                log.warn("Impossible de parser full_payload : {}", e.getMessage());
            }
        }
        return p;
    }

    /**
     * Construit le payload du rapport complet.
     *
     * Les données commerciales (cross-validation inter-documents) et la cohérence
     * LC/MT700 ↔ documents ne vivent PAS dans SwiftAnalysis.fullPayload (qui ne
     * contient que le MT700) : elles proviennent de la CommercialValidation liée
     * (passée via commercialValidationId) et d'un appel à la validation croisée LC.
     */
    private Map<String, Object> buildFullPayload(SwiftAnalysis a,
                                                 Long commercialValidationId,
                                                 String username) {
        Map<String, Object> p = buildMt700Payload(a);

        if (commercialValidationId == null) {
            log.info("Rapport complet sans commercialValidationId : sections commerciales omises.");
            return p;
        }

        Long userId = resolveUserId(username);
        CommercialValidation cv = commercialValidationRepo
                .findByIdAndUserId(commercialValidationId, userId)
                .orElse(null);
        if (cv == null) {
            log.warn("CommercialValidation {} introuvable pour l'utilisateur {} — sections commerciales omises.",
                    commercialValidationId, username);
            return p;
        }

        // ── Documents commerciaux + cohérence inter-documents ──────────────
        try {
            if (cv.getFullPayload() != null) {
                JsonNode full = objectMapper.readTree(cv.getFullPayload());
                p.put("commercial_documents", buildCommercialDocuments(full.path("documents")));
            }
            Map<String, Object> commVal = new HashMap<>();
            commVal.put("risk_level",  cv.getRiskLevel());
            commVal.put("final_score", cv.getFinalScore());
            p.put("commercial_validation", commVal);

            if (cv.getCrossReport() != null) {
                p.put("cross_validation_report",
                        buildCrossValidationReport(objectMapper.readTree(cv.getCrossReport())));
            }
        } catch (Exception e) {
            log.warn("Impossible de parser la CommercialValidation {} : {}",
                    commercialValidationId, e.getMessage());
        }

        // ── Cohérence LC/MT700 ↔ documents (recalcul à la volée) ───────────
        try {
            JsonNode coherence = lcCrossService.rawCoherence(a.getId(), commercialValidationId, username);
            p.put("lc_coherence", buildLcCoherence(coherence));
        } catch (Exception e) {
            log.warn("Cohérence LC/MT700 indisponible pour analyse {} : {}",
                    a.getId(), e.getMessage());
        }

        return p;
    }

    /** Mappe les documents du pipeline commercial vers la forme attendue par le template. */
    private List<Map<String, Object>> buildCommercialDocuments(JsonNode docs) {
        List<Map<String, Object>> out = new ArrayList<>();
        if (docs == null || !docs.isArray()) return out;
        for (JsonNode d : docs) {
            Map<String, Object> m = new HashMap<>();
            m.put("doc_type",        d.path("doc_type").asText(null));
            m.put("source_filename", d.path("source").asText(d.path("source_filename").asText("—")));
            // Le template colore selon 'OK'/'ERROR' ; le pipeline renvoie 'success'/'error'.
            String st = d.path("status").asText("error");
            m.put("status", "success".equals(st) ? "OK" : "error".equals(st) ? "ERROR" : st);
            m.put("n_fields",   d.path("fields").size());
            m.put("n_pages",    d.path("n_pages").asInt(0));
            m.put("n_warnings", d.path("warnings").size());
            JsonNode conf = d.path("confidence");
            m.put("confidence", (conf.isMissingNode() || conf.isNull()) ? 0.0 : conf.asDouble());

            int fails = 0;
            JsonNode rr = d.path("rule_results");
            if (rr.isArray()) {
                for (JsonNode r : rr) if ("fail".equals(r.path("status").asText())) fails++;
            } else {
                fails = d.path("rule_summary").path("fail").asInt(0);
            }
            m.put("n_rule_fails", fails);
            out.add(m);
        }
        return out;
    }

    /** cross_report (rule_results) → {risk_level, final_score, inconsistencies[]} attendu par le template. */
    private Map<String, Object> buildCrossValidationReport(JsonNode cross) {
        Map<String, Object> m = new HashMap<>();
        if (cross == null || cross.isMissingNode()) return m;
        String risk = cross.path("risk_level").asText(null);
        m.put("risk_level", risk != null ? risk.toUpperCase() : null);
        JsonNode fs = cross.path("final_score");
        m.put("final_score", (fs.isMissingNode() || fs.isNull()) ? 0.0 : fs.asDouble());

        List<Map<String, Object>> inconsistencies = new ArrayList<>();
        JsonNode rr = cross.path("rule_results");
        if (rr.isArray()) {
            for (JsonNode r : rr) {
                String st = r.path("status").asText("");
                if ("fail".equals(st) || "warning".equals(st)) {
                    Map<String, Object> inc = new HashMap<>();
                    inc.put("severity", r.path("severity").asText(st).toUpperCase());
                    inc.put("field",    r.path("field").asText(""));
                    inc.put("type",     r.path("rule").asText(r.path("doc_pair").asText("")));
                    inc.put("detail",   r.path("message").asText(""));
                    inconsistencies.add(inc);
                }
            }
        }
        m.put("inconsistencies", inconsistencies);
        return m;
    }

    /** Résultat LC/MT700 → {final_score, risk_level, decision, score_breakdown, rule_results(fail/warn)}. */
    @SuppressWarnings("unchecked")
    private Map<String, Object> buildLcCoherence(JsonNode coh) {
        Map<String, Object> m = new HashMap<>();
        if (coh == null || coh.isMissingNode()) return m;
        JsonNode fs = coh.path("final_score");
        m.put("final_score", (fs.isMissingNode() || fs.isNull()) ? 0.0 : fs.asDouble());
        String risk = coh.path("risk_level").asText(null);
        m.put("risk_level", risk != null ? risk.toUpperCase() : null);
        if (!coh.path("decision").isMissingNode() && !coh.path("decision").isNull()) {
            m.put("decision", coh.path("decision").asText());
        }

        Map<String, Object> breakdown = new LinkedHashMap<>();
        JsonNode bd = coh.path("score_breakdown");
        if (bd.isObject()) {
            bd.fields().forEachRemaining(e -> {
                try { breakdown.put(e.getKey(), e.getValue().asDouble()); } catch (Exception ignored) {}
            });
        }
        m.put("score_breakdown", breakdown);

        List<Map<String, Object>> rules = new ArrayList<>();
        JsonNode rr = coh.path("rule_results");
        if (rr.isArray()) {
            for (JsonNode r : rr) {
                String st = r.path("status").asText("");
                if ("fail".equals(st) || "warning".equals(st)) {
                    try { rules.add(objectMapper.convertValue(r, Map.class)); } catch (Exception ignored) {}
                }
            }
        }
        m.put("rule_results", rules);
        return m;
    }

    private Long toLong(Object v) {
        if (v == null) return null;
        if (v instanceof Number n) return n.longValue();
        try { return Long.parseLong(v.toString()); } catch (Exception e) { return null; }
    }
}
