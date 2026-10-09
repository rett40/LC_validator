# Swift Validator — Backend : Diagramme de classes complet

Application **Spring Boot 3 / Java (Jakarta)**, organisée en couches
`Controller → Service → Repository → Entity JPA`, avec sécurité JWT, stockage
MinIO (double client interne/public) et délégation à un **pipeline Python (FastAPI)**
pour toute la partie IA (OCR, NER SWIFT, conformité UCP, risque ML, validation croisée).

> Package racine : `com.swiftvalidator`
> Les diagrammes ci-dessous sont en [Mermaid](https://mermaid.js.org/).

---

## 1. Vue d'ensemble (couches & flux externes)

```mermaid
flowchart LR
    subgraph Client["Angular (4200)"]
    end
    subgraph Backend["Spring Boot — com.swiftvalidator"]
        CTRL["Controllers\n/api/v1/**"]
        SVC["Services\n@Service"]
        REPO["Repositories\nJpaRepository"]
        SEC["Security\nJWT + UserDetails"]
    end
    DB[("PostgreSQL\nJSONB")]
    MINIO[("MinIO\nobjets PDF/uploads")]
    PY["Pipeline Python\nFastAPI /pipeline /commercial /lc-commercial /pdf"]

    Client -->|cookie swift_auth / Bearer| CTRL
    CTRL --> SVC
    SVC --> REPO --> DB
    SVC -->|RestTemplate| PY
    SVC -->|MinioClient| MINIO
    SEC -.filtre.-> CTRL
```

---

## 2. Entités JPA (modèle de domaine)

```mermaid
classDiagram
    direction LR

    class User {
        +Long id
        +String username
        +String email
        +String passwordHash
        +String role = "AGENT"
        +Boolean enabled = true
        +LocalDateTime createdAt
        -onCreate()
    }

    class Document {
        +Long id
        +String originalFilename
        +String storedFilename
        +String contentType
        +Long fileSize
        +String minioBucket
        +String minioObjectKey
        +String checksum
        +LocalDateTime uploadedAt
    }

    class SwiftAnalysis {
        +Long id
        +String lcNumber
        +String swiftType
        +String lcForm
        +String applicant
        +String beneficiary
        +String issuingBank
        +String currency
        +BigDecimal amount
        +Integer tolerancePlus / toleranceMinus
        +LocalDate issueDate / expiryDate / shipmentDate
        +Integer presentationPeriodDays / validityDays
        +String incoterm / country
        +Boolean partialShipmentsAllowed / transhipmentAllowed
        +String goodsDescription / documentsRequired
        +Integer ucpComplianceScore / nlpScore / completenessScore / semanticScore
        +String riskLevel
        +BigDecimal riskScore / mlRiskScore
        +Boolean paymentBlocked
        +Integer nUcpErrors / nUcpWarnings / nUcpExclusions
        +Integer nClauses / nCriticalClauses / nHighRiskClauses
        +String rawText
        +String fullPayload  «jsonb»
        +String status = "PENDING"
        +String validationType = "MT700_ONLY"
        +Long processingMs
        +LocalDateTime createdAt / updatedAt
    }

    class UcpFinding {
        +Long id
        +String article / articleTitle / code / category / severity
        +Integer tier / cascadeImpact
        +BigDecimal weight / confidence
        +String message / clauseText / recommendation
        +Boolean blocksPayment
        +String findingNature / impactLevel / clauseType
    }

    class NlpClause {
        +Long id
        +String text / clauseType / importance / riskLevel
        +BigDecimal confidence
        +String method / ucpRefs
    }

    class DsRiskScore {
        +Long id
        +String riskLevel / modelUsed
        +BigDecimal riskScore / mlRiskScore
        +Integer completenessScore / nClauses / nCriticalClauses
        +Integer incotermRisk / presentationDays / validityDays
        +Long processingMs
    }

    class ExternalValidation {
        +Long id
        +String riskLevel / approach
        +BigDecimal rawScore
        +String currencyCode
        +Boolean currencyValid / countrySanctioned
        +String countryIso2 / fatfList / sourcesConsulted
        +String detailsJson  «jsonb»
        +LocalDateTime processedAt
    }

    class CommercialValidation {
        +Long id
        +String lcReference
        +Long swiftAnalysisId
        +String status = "PENDING"
        +String riskLevel
        +BigDecimal finalScore
        +Integer nDocuments / nProcessed / nFailed
        +String crossReport  «jsonb»
        +String fullPayload  «jsonb»
        +LocalDateTime createdAt / updatedAt
    }

    class CommercialDocument {
        +Long id
        +String sourceFilename / docType / inputType / status
        +BigDecimal confidence / extractionScore / processingTimeS
        +Integer nFields / nPages / nWarnings / nRuleFails
        +String fieldsJson / tablesJson / tiersJson / ruleResultsJson  «jsonb»
    }

    class AnalysisReport {
        +Long id
        +Long analysisId
        +String reportType  «MT700_REPORT|FULL_REPORT»
        +String displayName
        +String minioObjectKey
        +Long fileSize
        +LocalDateTime generatedAt
        +Long generatedBy
    }

    class AuditLog {
        +Long id
        +String action / entityType
        +Long entityId / userId
        +String username / ipAddress / details
        +LocalDateTime createdAt
    }

    User "1" --> "0..*" SwiftAnalysis : analyses
    User "1" --> "0..*" Document : uploads
    User "1" --> "0..*" CommercialValidation

    SwiftAnalysis "1" --> "0..1" Document : OneToOne
    SwiftAnalysis "1" *-- "0..*" UcpFinding : ucpFindings
    SwiftAnalysis "1" *-- "0..*" NlpClause : nlpClauses
    SwiftAnalysis "1" *-- "0..1" ExternalValidation
    SwiftAnalysis "1" *-- "0..1" DsRiskScore

    CommercialValidation "1" *-- "0..*" CommercialDocument : documents
    CommercialDocument "1" --> "0..1" Document : OneToOne

    AnalysisReport ..> SwiftAnalysis : analysisId (lien lâche, pas de FK)
    CommercialValidation ..> SwiftAnalysis : swiftAnalysisId (lien lâche)
```

> Toutes les entités sont annotées `@Entity @Data @Builder @NoArgsConstructor @AllArgsConstructor` (Lombok).
> `AnalysisReport.analysisId` et `CommercialValidation.swiftAnalysisId` référencent `SwiftAnalysis` par **id brut** (pas de relation JPA) — dépendance volontairement lâche.

---

## 3. Couche applicative (Controllers, Services, Repositories)

```mermaid
classDiagram
    direction TB

    %% ───────── CONTROLLERS ─────────
    class AuthController {
        +login(LoginRequest, resp) AuthResponse
        +logout(resp) void
        +me(UserDetails) AuthResponse
    }
    class SwiftAnalysisController {
        +upload(MultipartFile, UserDetails) AnalysisResponseDTO
        +getById(Long, UserDetails)
        +list(page,size,riskLevel,lcNumber,UserDetails) Page
        +rawPayload(Long, UserDetails)
    }
    class CommercialValidationController {
        +validate(List~MultipartFile~, lcRef, swiftAnalysisId, UserDetails)
        +getById(Long, UserDetails)
        +list(page,size,riskLevel,lcRef,UserDetails) Page
    }
    class LcCrossValidationController {
        +validate(swiftAnalysisId, commercialValidationId, UserDetails)
    }
    class ReportController {
        +generateMt700(Long, UserDetails)
        +generateFull(Long, commercialValidationId, UserDetails)
        +listReports(Long)
        +refreshUrl(Long)
        +adminAllReports(type)
    }
    class AdminController {
        +getStats()
        +getAgents()
        +createAgent(CreateAgentRequest)
        +updateAgent(Long, UpdateAgentRequest)
        +toggleStatus(Long, ToggleStatusRequest)
        +deleteAgent(Long)
        +getAgentHistory(Long)
    }
    class DashboardController {
        +stats(UserDetails) DashboardStatsDTO
    }
    class IntegrationHealthController {
        +integrationHealth() Map
        -dbCheck() / minioCheck() / pythonCheck()
    }
    class GlobalExceptionHandler
    class ApiExceptionHandler

    %% ───────── SERVICES ─────────
    class AnalysisService {
        +analyzeFile(MultipartFile, username) AnalysisResponseDTO
        +getById(Long, username)
        +listByUser(username, Pageable)
        +search(username, riskLevel, lcNumber, Pageable)
        +getDashboardStats(username) DashboardStatsDTO
        -buildAnalysis(JsonNode, User, Document) SwiftAnalysis
    }
    class CommercialValidationService {
        +validate(files, username, lcRef)
        +getById / list / search
        +toDTO(CommercialValidation, includePayload)
        -buildValidation(JsonNode, User, files, lcRef)
    }
    class LcCrossValidationService {
        +validate(swiftAnalysisId, commercialValidationId, username)
        +rawCoherence(...) JsonNode
        -buildLcPayload / buildCommercialPayload
    }
    class ReportService {
        +generateMt700Report(analysisId, username)
        +generateFullReport(analysisId, commercialValidationId, username)
        +listReports(analysisId)
        +freshDownloadUrl(reportId)
        -callPythonPdf(path, payload)
        -presignedUrl(objectKey)
    }
    class DocumentStorageService {
        +store(MultipartFile, User) Document
        +getFile(objectKey) InputStream
        -ensureBucket()
    }
    class AdminService {
        +getStats() / getAgents() / createAgent()
        +updateAgent() / toggleStatus() / deleteAgent()
        +getAgentHistory(agentId)
    }
    class UserService {
        +findByUsername(username) Optional~User~
        +save(User)
    }
    class PipelineClientService {
        +analyze(MultipartFile) JsonNode
        +analyzeCommercialDocuments(files) JsonNode
        +callLcCommercialValidate(Map) JsonNode
        -validateContract(JsonNode)
    }
    class AnalysisMapper {
        +toDTO(SwiftAnalysis, includeRawPayload) AnalysisResponseDTO
        -extractRiskMl / extractSwiftValidation / extractSemanticValidation ...
    }

    %% ───────── REPOSITORIES (JpaRepository) ─────────
    class UserRepository {
        <<interface>>
        findByUsername / findByEmail
        existsByUsername / existsByEmail
        findByRoleNot / countActiveAgents
    }
    class SwiftAnalysisRepository {
        <<interface>>
        findByIdAndUserId / search
        countByUserId / countSince / avgUcpScore
        countByRiskLevelIn / countByDaySince
    }
    class CommercialValidationRepository {
        <<interface>>
        findByIdAndUserId / search / countByUserId
    }
    class DocumentRepository {
        <<interface>>
    }
    class NlpClauseRepository {
        <<interface>>
    }
    class UcpFindingRepository {
        <<interface>>
    }
    class AnalysisReportRepository {
        <<interface>>
        findByAnalysisIdOrderByGeneratedAtDesc
    }
    class AuditLogRepository {
        <<interface>>
    }

    %% ── Controller → Service ──
    AuthController ..> JwtTokenProvider
    AuthController ..> UserRepository
    SwiftAnalysisController ..> AnalysisService
    CommercialValidationController ..> CommercialValidationService
    LcCrossValidationController ..> LcCrossValidationService
    ReportController ..> ReportService
    AdminController ..> AdminService
    DashboardController ..> AnalysisService
    IntegrationHealthController ..> PipelineClientService

    %% ── Service → Service / Repo ──
    AnalysisService ..> PipelineClientService
    AnalysisService ..> DocumentStorageService
    AnalysisService ..> AnalysisMapper
    AnalysisService ..> SwiftAnalysisRepository
    AnalysisService ..> UcpFindingRepository
    AnalysisService ..> NlpClauseRepository
    AnalysisService ..> UserRepository
    AnalysisService ..> AuditLogRepository

    CommercialValidationService ..> PipelineClientService
    CommercialValidationService ..> DocumentStorageService
    CommercialValidationService ..> CommercialValidationRepository
    CommercialValidationService ..> DocumentRepository
    CommercialValidationService ..> UserRepository
    CommercialValidationService ..> AuditLogRepository

    LcCrossValidationService ..> PipelineClientService
    LcCrossValidationService ..> SwiftAnalysisRepository
    LcCrossValidationService ..> CommercialValidationRepository
    LcCrossValidationService ..> UserRepository

    ReportService ..> AnalysisReportRepository
    ReportService ..> SwiftAnalysisRepository
    ReportService ..> CommercialValidationRepository
    ReportService ..> UserRepository
    ReportService ..> LcCrossValidationService

    AdminService ..> UserRepository
    AdminService ..> SwiftAnalysisRepository
    UserService ..> UserRepository
```

---

## 4. Sécurité & Configuration

```mermaid
classDiagram
    direction LR

    class JwtTokenProvider {
        -String jwtSecret
        -long jwtExpirationMs
        +generateToken(Authentication) String
        +getUsernameFromToken(token) String
        +validateToken(token) boolean
        -getKey() SecretKey
    }
    class JwtAuthFilter {
        +doFilterInternal(req, resp, chain)
        -extractToken(req) String
    }
    class CustomUserDetailsService {
        <<UserDetailsService>>
        +loadUserByUsername(username) UserDetails
    }
    class SecurityConfig {
        <<Configuration>>
        +filterChain(HttpSecurity) SecurityFilterChain
        +authProvider() DaoAuthenticationProvider
        +authenticationManager() AuthenticationManager
        +passwordEncoder() BCryptPasswordEncoder
        +corsSource() CorsConfigurationSource
    }
    class MinioConfig {
        <<Configuration>>
        +minioClient() MinioClient «@Primary, interne»
        +presignMinioClient() MinioClient «public»
    }
    class PipelineClientConfig {
        <<Configuration>>
        +restTemplate(builder) RestTemplate
    }
    class JwtConfig {
        <<Configuration>>
    }
    class DataInitializer {
        <<CommandLineRunner>>
        +run(args) : crée/maj admin
    }
    class SwiftValidatorApplication {
        <<@SpringBootApplication>>
        +main(String[])
    }
    class OncePerRequestFilter {
        <<abstract>>
    }

    JwtAuthFilter ..> JwtTokenProvider
    JwtAuthFilter ..> CustomUserDetailsService
    CustomUserDetailsService ..> UserRepository
    SecurityConfig ..> JwtAuthFilter
    SecurityConfig ..> CustomUserDetailsService
    DataInitializer ..> UserRepository
    OncePerRequestFilter <|-- JwtAuthFilter
```

---

## 5. DTOs (conteneurs Lombok `@Builder` avec sous-types imbriqués)

| DTO racine | Sous-types imbriqués notables |
|---|---|
| **AnalysisResponseDTO** | UcpFindingDTO, NlpClauseDTO, ExternalValidationDTO, ExternalDetailsDTO, ExternalCountryDTO, ExternalCurrencyDTO, DsRiskScoreDTO, RiskMlDTO, ShapFeatureDTO, ShapWaterfallItemDTO, ScoreBreakdownDTO, ExplanationDTO, RuleContributionDTO, TopDimensionDTO, MissingFieldItemDTO, UcpDetailDTO, UcpViolationDTO, SwiftValidationDTO, SwiftFieldValidationDTO, SemanticValidationDTO, SemanticItemDTO, CrossFieldIssueDTO |
| **CommercialValidationResponseDTO** | SummaryDTO, DocumentResultDTO, RuleSummaryDTO, CrossReportDTO |
| **LcCrossValidationResponseDTO** | LcSummaryDTO, CommercialSummaryInfo |
| **AdminDTO** | AdminStatsResponse, AgentResponse, AnalysisHistoryItem, DayCount, DecisionSegment, CreateAgentRequest, UpdateAgentRequest, ToggleStatusRequest |
| **AuthDTO** | LoginRequest, RegisterRequest, AuthResponse |
| **ReportDTO** | GenerateRequest, ReportResponse |
| **DashboardStatsDTO** | (plat : totalAnalyses, criticalCount, … avgUcpScore) |

---

## 6. Points d'architecture clés

- **Couches strictes** : `Controller` (REST, validation des fichiers, `@AuthenticationPrincipal`) → `Service` (`@Transactional`) → `Repository` (Spring Data JPA) → `Entity`.
- **L'IA n'est pas dans le Java** : tout passe par `PipelineClientService` (RestTemplate) vers le pipeline Python `swift-validator-ia`. Le JSON brut est stocké tel quel dans les colonnes `fullPayload` / `crossReport` (`jsonb`) puis re-parsé à la demande par `AnalysisMapper` et les services.
- **Deux clients MinIO** : `minioClient` (interne, upload serveur) et `presignMinioClient` (`@Qualifier("presignMinioClient")`, endpoint public pour URLs présignées 7 jours).
- **Sécurité** : JWT stateless, lu depuis le cookie `swift_auth` **ou** l'en-tête `Bearer` ; rôles `ADMIN` / `AGENT` ; `/api/v1/admin/**` protégé par `hasRole('ADMIN')` ; admin auto-créé au démarrage par `DataInitializer`.
- **Liens lâches volontaires** : `AnalysisReport.analysisId` et `CommercialValidation.swiftAnalysisId` ne sont pas des relations JPA mais des références par id.
