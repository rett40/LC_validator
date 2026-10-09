package com.swiftvalidator.controller;

import io.minio.BucketExistsArgs;
import io.minio.MinioClient;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/system")
@RequiredArgsConstructor
public class IntegrationHealthController {

    private final JdbcTemplate jdbcTemplate;
    private final MinioClient minioClient;
    private final RestTemplate restTemplate;

    @Value("${minio.bucket-name}")
    private String bucketName;

    @Value("${pipeline.base-url}")
    private String pipelineBaseUrl;

    @GetMapping("/integration-health")
    public ResponseEntity<Map<String, Object>> integrationHealth() {
        Map<String, Object> checks = new HashMap<>();
        checks.put("database", dbCheck());
        checks.put("minio", minioCheck());
        checks.put("pythonPipeline", pythonCheck());

        boolean up = checks.values().stream()
                .map(v -> (Map<?, ?>) v)
                .allMatch(m -> Boolean.TRUE.equals(m.get("up")));

        return ResponseEntity.ok(Map.of(
                "status", up ? "UP" : "DEGRADED",
                "checks", checks
        ));
    }

    private Map<String, Object> dbCheck() {
        try {
            Integer one = jdbcTemplate.queryForObject("SELECT 1", Integer.class);
            return Map.of("up", one != null && one == 1);
        } catch (Exception ex) {
            return Map.of("up", false, "error", ex.getMessage());
        }
    }

    private Map<String, Object> minioCheck() {
        try {
            boolean exists = minioClient.bucketExists(BucketExistsArgs.builder().bucket(bucketName).build());
            return Map.of("up", exists, "bucket", bucketName);
        } catch (Exception ex) {
            return Map.of("up", false, "error", ex.getMessage());
        }
    }

    private Map<String, Object> pythonCheck() {
        try {
            String body = restTemplate.getForObject(pipelineBaseUrl + "/health", String.class);
            return Map.of("up", body != null);
        } catch (Exception ex) {
            return Map.of("up", false, "error", ex.getMessage());
        }
    }
}
