package com.swiftvalidator.controller;

import com.swiftvalidator.dto.LcCrossValidationResponseDTO;
import com.swiftvalidator.service.LcCrossValidationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/lc-cross-validation")
@RequiredArgsConstructor
@Slf4j
public class LcCrossValidationController {

    private final LcCrossValidationService service;

    /**
     * Validates an already-analysed MT700 against already-validated commercial documents.
     *
     * POST /api/v1/lc-cross-validation
     *   ?swiftAnalysisId=42
     *   &commercialValidationId=7
     */
    @PostMapping
    public ResponseEntity<LcCrossValidationResponseDTO> validate(
            @RequestParam Long swiftAnalysisId,
            @RequestParam Long commercialValidationId,
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        if (swiftAnalysisId == null || commercialValidationId == null) {
            throw new IllegalArgumentException(
                    "swiftAnalysisId and commercialValidationId are required");
        }
        try {
            LcCrossValidationResponseDTO result = service.validate(
                    swiftAnalysisId, commercialValidationId, userDetails.getUsername());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            log.error("LC cross-validation failed: swiftAnalysisId={}, commercialValidationId={}",
                    swiftAnalysisId, commercialValidationId, e);
            throw new RuntimeException("LC cross-validation failed: " + e.getMessage(), e);
        }
    }
}
