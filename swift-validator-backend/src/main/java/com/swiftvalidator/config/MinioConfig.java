package com.swiftvalidator.config;

import io.minio.MinioClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

@Configuration
public class MinioConfig {

    @Value("${minio.endpoint}")
    private String endpoint;

    @Value("${minio.public-endpoint:${minio.endpoint}}")
    private String publicEndpoint;

    @Value("${minio.access-key}")
    private String accessKey;

    @Value("${minio.secret-key}")
    private String secretKey;

    /**
     * Région explicite : sans elle, le SDK appelle GetBucketLocation sur SON
     * endpoint avant la première opération (présignature incluse). Pour le
     * client presign pointé sur l'endpoint public (localhost:9000), cet appel
     * est impossible depuis le conteneur → presign cassé en prod.
     * (Même correctif que app/pdf/minio_storage.py côté Python.)
     */
    @Value("${minio.region:us-east-1}")
    private String region;

    /** Client interne (upload/download serveur→serveur) via le hostname Docker. */
    @Bean
    @Primary
    public MinioClient minioClient() {
        return MinioClient.builder()
            .endpoint(endpoint)
            .credentials(accessKey, secretKey)
            .region(region)
            .build();
    }

    /**
     * Client dédié à la GÉNÉRATION D'URLS PRÉSIGNÉES.
     * Signe avec l'endpoint PUBLIC (ex: http://localhost:9000) pour que le
     * navigateur puisse télécharger — la signature inclut le host, donc elle
     * doit correspondre à l'URL réellement utilisée par le client.
     */
    @Bean(name = "presignMinioClient")
    public MinioClient presignMinioClient() {
        return MinioClient.builder()
            .endpoint(publicEndpoint)
            .credentials(accessKey, secretKey)
            .region(region)
            .build();
    }
}
