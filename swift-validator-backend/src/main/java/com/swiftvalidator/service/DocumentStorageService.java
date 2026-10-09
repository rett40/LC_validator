package com.swiftvalidator.service;

import com.swiftvalidator.entity.Document;
import com.swiftvalidator.entity.User;
import io.minio.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class DocumentStorageService {

    @Value("${minio.bucket-name}")
    private String bucketName;

    @Value("${minio.enabled:false}")
    private boolean minioEnabled;

    private final MinioClient minioClient;

    public Document store(MultipartFile file, User user) throws Exception {
        byte[] bytes = file.getBytes();
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        String checksum = HexFormat.of().formatHex(md.digest(bytes));

        String objectKey = "uploads/" + user.getId() + "/" + UUID.randomUUID() + "_" + file.getOriginalFilename();

        if (minioEnabled) {
            ensureBucket();
            minioClient.putObject(
                PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectKey)
                    .stream(file.getInputStream(), file.getSize(), -1)
                    .contentType(file.getContentType())
                    .build()
            );
            log.info("Stored document {} -> MinIO key: {}", file.getOriginalFilename(), objectKey);
        } else {
            log.warn("MinIO disabled. Skipping object upload for {}", file.getOriginalFilename());
        }

        return Document.builder()
            .originalFilename(file.getOriginalFilename())
            .storedFilename(minioEnabled ? objectKey : file.getOriginalFilename())
            .contentType(file.getContentType())
            .fileSize(file.getSize())
            .minioBucket(minioEnabled ? bucketName : null)
            .minioObjectKey(minioEnabled ? objectKey : null)
            .checksum(checksum)
            .user(user)
            .build();
    }

    public InputStream getFile(String objectKey) throws Exception {
        return minioClient.getObject(
            GetObjectArgs.builder()
                .bucket(bucketName)
                .object(objectKey)
                .build()
        );
    }

    private void ensureBucket() throws Exception {
        boolean exists = minioClient.bucketExists(BucketExistsArgs.builder().bucket(bucketName).build());
        if (!exists) {
            minioClient.makeBucket(MakeBucketArgs.builder().bucket(bucketName).build());
        }
    }
}
