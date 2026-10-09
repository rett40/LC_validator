package com.swiftvalidator.dto;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

public class ReportDTO {

    @Data
    @Builder
    public static class ReportResponse {
        private Long          id;
        private Long          analysisId;
        private String        reportType;
        private String        displayName;
        private Long          fileSize;
        private LocalDateTime generatedAt;
        /** URL présignée valide 7 jours — régénérée à chaque appel */
        private String        downloadUrl;
    }

    @Data
    @Builder
    public static class GenerateRequest {
        /** ID de l'analyse Swift ou de la validation commerciale */
        private Long analysisId;
    }
}
