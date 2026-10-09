package com.swiftvalidator.controller;

import com.swiftvalidator.dto.AnalysisResponseDTO;
import com.swiftvalidator.service.AnalysisService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.net.URLConnection;
import java.util.List;

@RestController
@RequestMapping("/api/v1/analyses")
@RequiredArgsConstructor
@Slf4j
public class SwiftAnalysisController {

    private final AnalysisService analysisService;

    @PostMapping("/upload")
    public ResponseEntity<AnalysisResponseDTO> upload(
        @RequestParam("file") MultipartFile file,
        @AuthenticationPrincipal UserDetails userDetails
    ) {
        validateFile(file);
        try {
            AnalysisResponseDTO result = analysisService.analyzeFile(file, userDetails.getUsername());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            log.error("Analysis failed", e);
            throw new RuntimeException("Analysis failed: " + e.getMessage(), e);
        }
    }

    @GetMapping("/{id}")
    public ResponseEntity<AnalysisResponseDTO> getById(
        @PathVariable Long id,
        @AuthenticationPrincipal UserDetails userDetails
    ) {
        return ResponseEntity.ok(analysisService.getById(id, userDetails.getUsername()));
    }

    @GetMapping
    public ResponseEntity<Page<AnalysisResponseDTO>> list(
        @RequestParam(defaultValue = "0") int page,
        @RequestParam(defaultValue = "20") int size,
        @RequestParam(required = false) String riskLevel,
        @RequestParam(required = false) String lcNumber,
        @AuthenticationPrincipal UserDetails userDetails
    ) {
        PageRequest pr = PageRequest.of(page, size, Sort.by("createdAt").descending());

        if (riskLevel != null || lcNumber != null) {
            return ResponseEntity.ok(analysisService.search(userDetails.getUsername(), riskLevel, lcNumber, pr));
        }

        return ResponseEntity.ok(analysisService.listByUser(userDetails.getUsername(), pr));
    }

    @GetMapping("/{id}/raw-payload")
    public ResponseEntity<String> rawPayload(
        @PathVariable Long id,
        @AuthenticationPrincipal UserDetails userDetails
    ) {
        AnalysisResponseDTO dto = analysisService.getById(id, userDetails.getUsername());
        return ResponseEntity.ok("{\"id\": " + id + "}");
    }

    private void validateFile(MultipartFile file) {
        if (file.isEmpty()) {
            throw new IllegalArgumentException("File is empty");
        }

        String ct = detectedContentType(file);
        List<String> allowed = List.of("application/pdf", "image/png", "image/jpeg", "image/jpg", "image/tiff", "image/webp");

        if (ct == null || !allowed.contains(ct)) {
            throw new IllegalArgumentException("Unsupported file type: " + ct + ". Allowed: PDF, PNG, JPEG, TIFF, WEBP");
        }

        if (file.getSize() > 50 * 1024 * 1024) {
            throw new IllegalArgumentException("File too large (max 50MB)");
        }
    }

    private String detectedContentType(MultipartFile file) {
        String fromHeader = file.getContentType();
        try (InputStream stream = file.getInputStream()) {
            String fromStream = URLConnection.guessContentTypeFromStream(stream);
            return fromStream != null ? fromStream : fromHeader;
        } catch (IOException e) {
            return fromHeader;
        }
    }
}
