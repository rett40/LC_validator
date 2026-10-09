package com.swiftvalidator.repository;

import com.swiftvalidator.entity.Document;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface DocumentRepository extends JpaRepository<Document, Long> {
    // CORRECTION : utilisez le nom exact du champ dans l'entité
    Optional<Document> findByOriginalFilename(String originalFilename);
}