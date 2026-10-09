package com.swiftvalidator.controller;

import com.swiftvalidator.dto.ReportDTO;
import com.swiftvalidator.service.ReportService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/reports")
@RequiredArgsConstructor
public class ReportController {

    private final ReportService reportService;

    /**
     * Génère le rapport MT700 pour une analyse.
     * Si déjà généré, génère une nouvelle version.
     */
    @PostMapping("/mt700/{analysisId}")
    public ResponseEntity<ReportDTO.ReportResponse> generateMt700(
            @PathVariable Long analysisId,
            @AuthenticationPrincipal UserDetails userDetails) {
        ReportDTO.ReportResponse report = reportService.generateMt700Report(
            analysisId, userDetails != null ? userDetails.getUsername() : null);
        return ResponseEntity.status(201).body(report);
    }

    /**
     * Génère le rapport complet (MT700 + docs commerciaux) pour une analyse.
     */
    @PostMapping("/full/{analysisId}")
    public ResponseEntity<ReportDTO.ReportResponse> generateFull(
            @PathVariable Long analysisId,
            @RequestParam(required = false) Long commercialValidationId,
            @AuthenticationPrincipal UserDetails userDetails) {
        ReportDTO.ReportResponse report = reportService.generateFullReport(
            analysisId, commercialValidationId,
            userDetails != null ? userDetails.getUsername() : null);
        return ResponseEntity.status(201).body(report);
    }

    /**
     * Liste tous les rapports d'une analyse avec leurs URLs présignées fraîches.
     */
    @GetMapping("/{analysisId}")
    public List<ReportDTO.ReportResponse> listReports(@PathVariable Long analysisId) {
        return reportService.listReports(analysisId);
    }

    /**
     * Régénère une URL présignée fraîche (7 jours) pour un rapport existant.
     * Utile quand l'ancienne URL a expiré.
     */
    @GetMapping("/refresh/{reportId}")
    public ResponseEntity<ReportDTO.ReportResponse> refreshUrl(@PathVariable Long reportId) {
        String url = reportService.freshDownloadUrl(reportId);
        return ResponseEntity.ok(ReportDTO.ReportResponse.builder()
            .id(reportId).downloadUrl(url).build());
    }

    // ── Admin ──────────────────────────────────────────────────────────────

    /**
     * Admin : liste tous les rapports de toutes les analyses.
     */
    @GetMapping("/admin/all")
    @PreAuthorize("hasRole('ADMIN')")
    public List<ReportDTO.ReportResponse> adminAllReports(
            @RequestParam(required = false) String type) {
        // TODO : ajouter pagination et filtre par type
        return List.of();
    }
}
