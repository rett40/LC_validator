package com.swiftvalidator.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class PipelineClientService {

    @Value("${pipeline.base-url}")
    private String pipelineBaseUrl;

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public JsonNode analyze(MultipartFile file) throws IOException {
        // The Python pipeline expects file uploads on /pipeline (not /analyze which is text-only)
        String url = pipelineBaseUrl + "/pipeline";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

        ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        };

        body.add("file", resource);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        log.info("Calling pipeline at {} for file: {}", url, file.getOriginalFilename());
        long t0 = System.currentTimeMillis();

        ResponseEntity<String> response;
        try {
            response = restTemplate.postForEntity(url, requestEntity, String.class);
        } catch (RestClientException ex) {
            throw new RuntimeException("Python pipeline unavailable: " + ex.getMessage(), ex);
        }

        log.info("Pipeline responded in {}ms — status: {}", System.currentTimeMillis() - t0, response.getStatusCode());

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Pipeline error: " + response.getStatusCode());
        }

        JsonNode payload = objectMapper.readTree(response.getBody());
        validateContract(payload);
        return payload;
    }

    public JsonNode analyzeCommercialDocuments(List<MultipartFile> files) throws IOException {
        String url = pipelineBaseUrl + "/commercial/validate";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

        for (MultipartFile file : files) {
            ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            };
            body.add("files", resource);
        }

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        log.info("Calling commercial pipeline at {} for {} file(s)", url, files.size());
        long t0 = System.currentTimeMillis();

        ResponseEntity<String> response;
        try {
            response = restTemplate.postForEntity(url, requestEntity, String.class);
        } catch (RestClientException ex) {
            throw new RuntimeException("Python commercial pipeline unavailable: " + ex.getMessage(), ex);
        }

        log.info("Commercial pipeline responded in {}ms — status: {}",
                System.currentTimeMillis() - t0, response.getStatusCode());

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Commercial pipeline error: " + response.getStatusCode());
        }

        JsonNode payload = objectMapper.readTree(response.getBody());
        if (!payload.path("success").asBoolean(true)) {
            String message = payload.path("error").asText("Commercial pipeline returned failure");
            throw new RuntimeException(message);
        }
        return payload;
    }

    public JsonNode callLcCommercialValidate(Map<String, Object> body) throws IOException {
        String url = pipelineBaseUrl + "/lc-commercial/validate";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        String jsonBody = objectMapper.writeValueAsString(body);
        HttpEntity<String> requestEntity = new HttpEntity<>(jsonBody, headers);

        log.info("Calling LC-Commercial validation at {}", url);
        long t0 = System.currentTimeMillis();

        ResponseEntity<String> response;
        try {
            response = restTemplate.postForEntity(url, requestEntity, String.class);
        } catch (RestClientException ex) {
            throw new RuntimeException("LC-Commercial validator unavailable: " + ex.getMessage(), ex);
        }

        log.info("LC-Commercial validation responded in {}ms — status: {}",
                System.currentTimeMillis() - t0, response.getStatusCode());

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("LC-Commercial validation error: " + response.getStatusCode());
        }

        JsonNode payload = objectMapper.readTree(response.getBody());
        if (!payload.path("success").asBoolean(true)) {
            throw new RuntimeException(
                    payload.path("error").asText("LC-Commercial validation returned failure"));
        }
        return payload;
    }

    private void validateContract(JsonNode payload) {
        if (!payload.path("success").asBoolean(true)) {
            String message = payload.path("message").asText("");
            if (message.isBlank()) {
                message = payload.path("error").asText("");
            }
            if (message.isBlank()) {
                message = "Pipeline returned failure";
            }
            throw new RuntimeException(message);
        }
        String[] requiredTopLevel = {"ocr", "swift", "ucp_compliance", "ds_analysis", "external_validation"};
        for (String key : requiredTopLevel) {
            if (payload.path(key).isMissingNode() || payload.path(key).isNull()) {
                throw new RuntimeException("Invalid pipeline contract: missing '" + key + "'");
            }
        }
        if (payload.path("swift").path("entities").isMissingNode()) {
            throw new RuntimeException("Invalid pipeline contract: missing 'swift.entities'");
        }
        String contractVersion = payload.path("contract_version").asText("");
        if (!contractVersion.startsWith("1.")) {
            throw new RuntimeException("Unsupported pipeline contract_version: " + contractVersion);
        }
    }
}
