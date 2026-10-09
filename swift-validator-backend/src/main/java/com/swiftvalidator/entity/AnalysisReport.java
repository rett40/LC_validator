package com.swiftvalidator.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "analysis_reports")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AnalysisReport {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "analysis_id", nullable = false)
    private Long analysisId;

    /** MT700_REPORT | FULL_REPORT */
    @Column(name = "report_type", length = 20)
    private String reportType;

    /** Nom affiché à l'utilisateur */
    @Column(name = "display_name")
    private String displayName;

    /** Clé objet MinIO — jamais exposée directement */
    @Column(name = "minio_object_key", length = 500)
    private String minioObjectKey;

    @Column(name = "file_size")
    private Long fileSize;

    @Column(name = "generated_at")
    private LocalDateTime generatedAt;

    @Column(name = "generated_by")
    private Long generatedBy;

    @PrePersist
    protected void onCreate() {
        if (generatedAt == null) generatedAt = LocalDateTime.now();
    }
}
