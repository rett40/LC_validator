package com.swiftvalidator.controller;

import com.swiftvalidator.dto.CommercialValidationResponseDTO;
import com.swiftvalidator.service.CommercialValidationService;
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

import java.io.InputStream;
import java.net.URLConnection;
import java.util.List;
import java.util.Set;

@RestController
@RequestMapping("/api/v1/commercial")
@RequiredArgsConstructor
@Slf4j
public class CommercialValidationController {

    private static final Set<String> ALLOWED_TYPES = Set.of(
            "application/pdf", "image/png", "image/jpeg",
            "image/jpg", "image/tiff", "image/webp", "image/bmp"
    );

    private final CommercialValidationService service;

    @PostMapping("/validate")
    public ResponseEntity<CommercialValidationResponseDTO> validate(
            @RequestParam("files") List<MultipartFile> files,
            @RequestParam(value = "lcReference", required = false) String lcReference,
            @RequestParam(value = "swiftAnalysisId", required = false) Long swiftAnalysisId,
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        if (files == null || files.isEmpty()) {
            throw new IllegalArgumentException("Au moins un fichier est requis.");
        }
        if (files.size() > 3) {
            throw new IllegalArgumentException("Maximum 3 documents par appel.");
        }
        files.forEach(this::validateFile);

        try {
            CommercialValidationResponseDTO result =
                    service.validate(files, userDetails.getUsername(), lcReference, swiftAnalysisId);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            log.error("Commercial validation failed", e);
            throw new RuntimeException("Commercial validation failed: " + e.getMessage(), e);
        }
    }

    @GetMapping("/{id}")
    public ResponseEntity<CommercialValidationResponseDTO> getById(
            @PathVariable Long id,
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        return ResponseEntity.ok(service.getById(id, userDetails.getUsername()));
    }

    @GetMapping
    public ResponseEntity<Page<CommercialValidationResponseDTO>> list(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) String riskLevel,
            @RequestParam(required = false) String lcReference,
            @RequestParam(required = false) Long swiftAnalysisId,
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        PageRequest pr = PageRequest.of(page, size, Sort.by("createdAt").descending());
        // Historique aligné sur le MT700 courant : ne renvoyer que les validations
        // commerciales rattachées à cette analyse SWIFT.
        if (swiftAnalysisId != null) {
            return ResponseEntity.ok(
                    service.listByAnalysis(userDetails.getUsername(), swiftAnalysisId, pr)
            );
        }
        if (riskLevel != null || lcReference != null) {
            return ResponseEntity.ok(
                    service.search(userDetails.getUsername(), riskLevel, lcReference, pr)
            );
        }
        return ResponseEntity.ok(service.list(userDetails.getUsername(), pr));
    }

    private void validateFile(MultipartFile file) {
        if (file.isEmpty()) {
            throw new IllegalArgumentException("Fichier vide : " + file.getOriginalFilename());
        }
        String ct = detectContentType(file);
        if (ct == null || !ALLOWED_TYPES.contains(ct)) {
            throw new IllegalArgumentException(
                    "Type non supporté : " + ct + " (" + file.getOriginalFilename() + ")"
            );
        }
        if (file.getSize() > 50 * 1024 * 1024) {
            throw new IllegalArgumentException(
                    "Fichier trop volumineux (max 50 Mo) : " + file.getOriginalFilename()
            );
        }
    }

    private String detectContentType(MultipartFile file) {
        String fromHeader = file.getContentType();
        try (InputStream stream = file.getInputStream()) {
            String detected = URLConnection.guessContentTypeFromStream(stream);
            return detected != null ? detected : fromHeader;
        } catch (Exception e) {
            return fromHeader;
        }
    }
}
