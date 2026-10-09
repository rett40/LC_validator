export interface UcpFindingDTO {
  article: string;
  articleTitle: string;
  code: string;
  category: string;
  severity: 'error' | 'warning' | 'info';
  tier: number;
  weight?: number;
  confidence?: number;
  message: string;
  clauseText: string;
  recommendation: string;
  blocksPayment: boolean;
  cascadeImpact: number;
  exclusionType: string;
  /** true_violation | contractual_modification | operational_discrepancy */
  findingNature?: string;
  /** HIGH | MEDIUM | LOW */
  impactLevel?: string;
  /** financial | transport | authentication | identity | compliance | document_format */
  clauseType?: string;
  blocksPaymentReason?: string;
}

export interface NlpClauseDTO {
  text: string;
  clauseType: string;
  importance: string;
  riskLevel: string;
  confidence: number;
  method: string;
  ucpRefs: string;
}

export interface ExternalValidationDTO {
  riskLevel: 'CLEAN' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  rawScore: number;
  approach: string;
  currencyCode: string;
  currencyValid: boolean;
  countryIso2: string;
  countrySanctioned: boolean;
  fatfList: string;
  sourcesConsulted: string;
  processingMs: number;
  details?: ExternalDetailsDTO | null;
}

export interface ExternalCountryDTO {
  raw: string;
  iso2: string;
  risk: string;
  details: string;
}

export interface ExternalCurrencyDTO {
  code: string;
  isValid: boolean;
  risk: string;
  details: string;
}

export interface ExternalDetailsDTO {
  countries: ExternalCountryDTO[];
  currency: ExternalCurrencyDTO | null;
}

export interface DsRiskScoreDTO {
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  riskScore: number;
  mlRiskScore: number;
  modelUsed: string;
  completenessScore: number;
  nClauses: number;
  nCriticalClauses: number;
}

export interface ShapFeatureDTO {
  feature: string;
  shapValue: number;
  direction: string;
}

export interface UcpViolationDTO {
  article: string;
  field: string;
  description: string;
  severity: 'BLOCK' | 'HIGH' | 'WARN';
  overrideScore?: string;
}

// ── Score Breakdown v4.0 ───────────────────────────────────────────────────

export interface ScoreBreakdownDimension {
  score: number;
  weight: number;
  contribution: number;
  // completeness
  missingCriticalFields?: number;
  criticalFieldsPresentRate?: number;
  completenessRate?: number;
  // ucp_compliance
  errorsCount?: number;
  warningsCount?: number;
  weightedViolations?: number;
  complianceScorePct?: number;
  errorWeight?: number;
  warningWeight?: number;
  // coherence
  dateCoherencePct?: number;
  semanticCoherencePct?: number;
  contradictionDetected?: boolean;
  // country_risk
  applicantRisk?: number;
  beneficiaryRisk?: number;
  isSanctioned?: boolean;
  dualSanction?: boolean;
  // temporal
  validityDays?: number;
  daysToShipment?: number;
  presentationDays?: number;
  validityRisk?: number;
  shipmentRisk?: number;
  presentationRisk?: number;
  // amount
  tier?: number;
  lcAmountUsd?: number;
  redClause?: boolean;
  tenorRisk?: number;
  bankTier?: number;
  // ml_model
  label?: string;
  probabilities?: Record<string, number>;
}

export interface ScoreBreakdownDTO {
  analyticalScore: number;
  analyticalLabel: 'LOW' | 'MEDIUM' | 'HIGH';
  weights: Record<string, number>;
  dimensions: {
    completeness:   ScoreBreakdownDimension;
    ucp_compliance: ScoreBreakdownDimension;
    coherence:      ScoreBreakdownDimension;
    country_risk:   ScoreBreakdownDimension;
    temporal:       ScoreBreakdownDimension;
    amount:         ScoreBreakdownDimension;
    ml_model:       ScoreBreakdownDimension;
  };
}

export interface MissingFieldItemDTO {
  field: string;
  name: string;
  criticality: 'CRITICAL' | 'HIGH' | 'OPTIONAL';
  impactScore: number;
}

export interface ShapWaterfallItemDTO {
  feature: string;
  value: number | null;
  shapValue: number;
  contributionPct: number;
  direction: string;
}

export interface RuleContributionDTO {
  rule: string;
  field: string;
  severity: 'BLOCK' | 'HIGH' | 'WARN';
  description: string;
  scoreImpact: string;
}

export interface ExplanationDTO {
  method: string;
  finalLabel: string;
  riskScoreMl: number;
  analyticalScore: number;
  shapWaterfall: ShapWaterfallItemDTO[];
  dimensionRadar: Record<string, number>;
  ruleContributions: RuleContributionDTO[];
  topContributingDimensions: { dimension: string; contribution: number }[];
  topShapFeatures?: ShapFeatureDTO[];
}

export interface UcpDetailDTO {
  errorsCount: number;
  warningsCount: number;
  weightedViolationIndex: number;
  errorWeight: number;
  warningWeight: number;
  complianceScorePct: number;
  ucpDecision: string;
  paymentBlocked: boolean;
  ruleViolationsCount: number;
  blockViolations: Record<string, unknown>[];
  highViolations: Record<string, unknown>[];
  warnViolations: Record<string, unknown>[];
  errorsDetail: Record<string, unknown>[];
  warningsDetail: Record<string, unknown>[];
}

export interface RiskMlDTO {
  riskLabel: string;
  riskProba: Record<string, number>;
  riskScore: number;
  isAnomaly: boolean;
  anomalyScore: number;
  segment: string;
  segmentId: number;
  narrative: string;
  shapTopFeatures: ShapFeatureDTO[];
  ucpViolations?: UcpViolationDTO[];
  modelVersion: string;
  challengerLabel: string;
  processingMs: number;
  error: string;
  // v4.0 additions
  scoreBreakdown?: ScoreBreakdownDTO;
  missingFieldsDetail?: MissingFieldItemDTO[];
  explanation?: ExplanationDTO;
  ucpDetail?: UcpDetailDTO;
  finalRiskScore?: number;
  finalRiskScoreComponents?: Record<string, number>;
}

export interface SwiftFieldValidationDTO {
  tag: string;
  name: string;
  mandatory: boolean;
  valid: boolean;
  value: string | null;
  errors: string[];
  warnings: string[];
}

export interface SwiftValidationDTO {
  score: number;
  overallValid: boolean;
  summary: string;
  missingMandatory: string[];
  fields: SwiftFieldValidationDTO[];
  nErrors: number;
  nWarnings: number;
}

export interface SemanticItemDTO {
  code: string;
  severity: string;
  message: string;
  detail: string;
  field: string;
  relatedField: string;
}

export interface CrossFieldIssueDTO {
  fields: string[];
  code: string;
  technique: 'NLI' | 'NLI+regex' | 'regex' | 'rule';
  severity: 'error' | 'warning' | 'info';
  message: string;
  suggestion: string;
  lcValue: string;
  docValue: string;
  confidence: number;
  ucpRef: string;
}

export interface SemanticValidationDTO {
  completenessScore: number;
  semanticScore: number;
  bayesianConfidence: number;
  docQualityScore: number;
  summary: string;
  warnings: SemanticItemDTO[];
  contradictions: SemanticItemDTO[];
  riskFlags: SemanticItemDTO[];
  recommendations: string[];
  coherenceScoresJson: string;
  geoCoherenceJson: string;
  fraudIndicators: string[];

  // AXE 7 — Cross-field coherence
  crossFieldIssues?: CrossFieldIssueDTO[];
  crossFieldCoherenceScore?: number;
  crossFieldVerdict?: string;
}

export interface AnalysisResponseDTO {
  id: number;
  lcNumber: string;
  swiftType: string;
  lcForm: string;

  applicant: string;
  beneficiary: string;
  issuingBank: string;

  currency: string;
  amount: number;
  tolerancePlus: number;
  toleranceMinus: number;

  issueDate: string;
  expiryDate: string;
  shipmentDate: string;
  presentationPeriodDays: number;
  validityDays: number;

  incoterm: string;
  country: string;
  partialShipmentsAllowed: boolean | null;
  transhipmentAllowed: boolean | null;
  confirmationInstructions: string;
  applicableRules: string;
  goodsDescription: string;
  documentsRequired: string;
  additionalConditions?: string;

  ucpComplianceScore: number;
  nlpScore: number;
  completenessScore: number;
  semanticScore: number;

  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'CLEAN';
  riskScore: number;        // DS score 0-1 (legacy)
  mlRiskScore: number;      // DS ML score 0-1 (legacy)
  /** Score de risque final unifié 0-100 — source canonique pour l'affichage */
  finalRiskScore?: number;
  paymentBlocked: boolean;
  countryRisk: string;
  riskFactors?: string[];

  nUcpErrors: number;
  nUcpWarnings: number;
  nUcpExclusions: number;
  nClauses: number;
  nCriticalClauses: number;

  ucpFindings?: UcpFindingDTO[];
  nlpClauses?: NlpClauseDTO[];
  externalValidation?: ExternalValidationDTO | null;
  dsRiskScore?: DsRiskScoreDTO | null;
  riskMl?: RiskMlDTO;
  swiftValidation?: SwiftValidationDTO;
  semanticValidation?: SemanticValidationDTO;

  status: string;
  processingMs: number;
  createdAt: string;
  rawPayloadJson?: string | null;

  /** Type de validation — MT700 seule ou validation complète */
  validationType?: 'MT700_ONLY' | 'FULL_VALIDATION';

  // ── Rapports PDF (chargés côté UI) ───────────────────────────────
  reports?: ReportMetaDTO[];
  /** Flag UI transitoire pendant la génération d'un rapport */
  generatingReport?: boolean;
}

export interface ReportMetaDTO {
  id:          number;
  analysisId:  number;
  reportType:  'MT700_REPORT' | 'FULL_REPORT';
  displayName: string;
  fileSize:    number;
  generatedAt: string;
  downloadUrl: string;
}

export interface DashboardStatsDTO {
  totalAnalyses: number;
  criticalCount: number;
  highCount: number;
  mediumCount: number;
  cleanCount: number;
  last24hCount: number;
  paymentBlockedCount: number;
  avgUcpScore: number;
}

export interface AuthResponse {
  token?: string;
  tokenType: string;
  userId: number;
  username: string;
  role: string;
  expiresIn: number;
}

export interface RawAnalysisPayload {
  ocr?: RawOcrPayload;
  swift?: RawSwiftPayload;
  ucp_compliance?: RawUcpPayload;
  ds_analysis?: RawDsAnalysisPayload;
  external_validation?: RawExternalValidationPayload;
  risk_ml?: RawRiskMlPayload;
  /** Verdict final réconcilié (hard override UCP prioritaire sur le score composite) */
  final_decision?: RawFinalDecision;
}

export interface RawFinalDecision {
  /** ACCEPT | ACCEPT_WITH_FLAGS | REVIEW | REJECT */
  decision?: string;
  /** ucp_hard_block | ucp_contractual_hold | ucp_compliance | composite_router | fallback */
  source?: string;
  reason?: string;
  inputs?: {
    ucp_decision?: string | null;
    ucp_payment_blocked?: boolean;
    ucp_score?: number | null;
    composite_decision?: string | null;
    composite_score?: number | null;
  };
}

export interface RawOcrPayload {
  success?: boolean;
  text?: string;
  confidence?: number;
  [key: string]: unknown;
}

export interface RawSwiftPayload {
  success?: boolean;
  swift_type?: string;
  error?: string;
  entities?: RawSwiftEntities;
  validation?: RawValidationPayload;
  nlp?: RawNlpPayload;
  [key: string]: unknown;
}

export interface RawSwiftEntities {
  lc_form?: string;
  lc_number?: string;
  issue_date?: string;
  applicable_rules?: string;
  expiry_date?: string;
  applicant?: string;
  beneficiary?: string;
  currency?: string;
  amount?: number;
  tolerance_plus?: number;
  tolerance_minus?: number;
  available_with?: string;
  available_by?: string;
  draft_at?: string;
  partial_shipments?: string | boolean;
  transhipment?: string | boolean;
  port_loading?: string;
  port_discharge?: string;
  shipment_date?: string;
  presentation_period?: string | number;
  confirmation?: string;
  goods_description?: string;
  documents_required?: string;
  additional_conditions?: string;
  charges?: string;
  sender_info?: string;
  guarantee_text?: string;
  issuing_bank?: string;
  country?: string;
  incoterm?: string;
  raw_fields?: Record<string, string>;
  [key: string]: unknown;
}

export interface RawValidationPayload {
  overall_valid?: boolean;
  score?: number;
  summary?: string;
  missing_mandatory?: string[];
  cross_field_errors?: string[];
  cross_field_warnings?: string[];
  fields?: Record<string, RawValidationField>;
}

export interface RawValidationField {
  tag?: string;
  name?: string;
  mandatory?: boolean;
  valid?: boolean;
  value?: unknown;
  errors?: string[];
  warnings?: string[];
  confidence?: number;
}

export interface RawNlpPayload {
  score?: number;
  processing_ms?: number;
  modules_applied?: string[];
  module_timings?: Record<string, number>;
  llm_triggered?: boolean;
  from_cache?: boolean;
  spell?: RawSpellPayload | null;
  ner?: RawNerPayload | null;
  clauses?: RawClause[];
  contradictions?: RawContradiction[];
  semantic?: RawSemanticPayload | null;
}

export interface RawSpellPayload {
  total_changes?: number;
  has_ocr_noise?: boolean;
  language_detected?: string;
  avg_confidence?: number;
  rejected_count?: number;
  corrections?: RawCorrection[];
}

export interface RawCorrection {
  original?: string;
  corrected?: string;
  method?: string;
  position?: number;
  confidence?: number;
  validated?: boolean;
}

export interface RawNerPayload {
  entities?: RawNerEntity[];
  spacy_spans?: unknown[];
  source_mode?: string;
  relations?: RawRelation[];
}

export interface RawNerEntity {
  label?: string;
  value?: string;
  start?: number;
  end?: number;
  confidence?: number;
  source?: string;
  ner_label?: string;
  country?: string;
  normalized?: string;
}

export interface RawRelation {
  subject?: string;
  predicate?: string;
  object_label?: string;
  object_value?: string;
}

export interface RawClause {
  text?: string;
  type?: string;
  importance?: string;
  confidence?: number;
  method?: string;
  ucp_refs?: string[];
  risk_level?: string;
}

export interface RawContradiction {
  // Champs NLP (inter-clause)
  severity?: string;
  message?: string;
  clause_a?: string;
  clause_b?: string;
  source?: string;
  // Champs ContradictionEngine (tag vs clause)
  tag_field?: string;
  tag_value?: string;
  clause_text?: string;
  clause_intent?: string;
  recommendation?: string;
  ucp_basis?: string;
  confidence?: number;
  [key: string]: unknown;
}

export interface RawCrossFieldCoherence {
  coherence_score?: number;
  verdict?: string;
  axes_checked?: string[];
  processing_ms?: number;
  issues?: RawCrossFieldIssue[];
}

export interface RawCrossFieldIssue {
  fields?: string[];
  code?: string;
  technique?: string;
  severity?: string;
  message?: string;
  suggestion?: string;
  lc_value?: string;
  doc_value?: string;
  confidence?: number;
  ucp_ref?: string;
}

export interface RawSemanticPayload {
  completeness_score?: number;
  semantic_score?: number;
  doc_quality_score?: number;
  bayesian_confidence?: number;
  final_score?: number;
  routing_decision?: string;
  warnings?: RawSemanticIssue[];
  risk_flags?: RawRiskFlag[];
  contradictions?: RawContradiction[];
  fraud_indicators?: Array<string | RawFraudIndicator>;
  recommendations?: string[];
  summary?: string;
  enriched_fields?: Record<string, unknown>;
  coherence_scores?: Record<string, number>;
  geo_coherence?: boolean | RawGeoEntry[] | Record<string, unknown>;
  cross_field_coherence?: RawCrossFieldCoherence;
}

export interface RawSemanticIssue {
  code?: string;
  severity?: string;
  message?: string;
  detail?: string;
  field?: string;
  related_field?: string;
  ucp_ref?: string;
  suggestion?: string;
  source?: string;  // "rule" | "fuzzy" | "bayesian" | "ontology" | "nli"
  confidence?: number;
}

export interface RawRiskFlag {
  code?: string;
  description?: string;
  severity?: string;
  category?: string;
  ucp_ref?: string;
  ml_score?: number;
}

export interface RawFraudIndicator {
  code?: string;
  description?: string;
  severity?: string;
  score?: number;
}

export interface RawGeoEntry {
  status?: string;
  country?: string;
  port?: string;
  risk?: string;
  details?: string;
  [key: string]: unknown;
}

// ─── UCP interfaces ────────────────────────────────────────────────────────────

export interface RawUcpBlockingViolation {
  article?: string;
  article_title?: string;
  code?: string;
  reason?: string;
  description?: string;
  clause_text?: string;
  recommendation?: string;
  [key: string]: unknown;
}

export interface RawUcpPayload {
  global_score?: number;
  compliant?: boolean;
  findings?: RawUcpFinding[];
  summary?: RawUcpSummary;
  categories?: Record<string, RawUcpCategory>;
  exclusions?: Record<string, string>;
  processing_ms?: number;
  payment_blocked?: boolean;
  /** true when a contractual_modification finding blocks payment (Art.1 derogation held) */
  contractual_hold?: boolean;
  /** ACCEPT | ACCEPT_WITH_FLAGS | REVIEW | CONDITIONAL_HOLD | REJECT */
  ucp_decision?: string | { decision?: string };
  /** Documentary risk level: LOW | MEDIUM | HIGH | CRITICAL */
  contractual_risk_level?: string;
  documentary_discrepancy_risk?: string;
  semantic?: RawUcpSemanticMeta;
  /** UCP version string displayed in the panel header, e.g. "UCP 600" */
  ucp_version?: string;
  /** Human-readable legal compliance status label */
  legal_compliance_status?: string;
  /** Violations that directly block payment */
  ucp_blocking_violations?: RawUcpBlockingViolation[];
}

export interface RawUcpFinding {
  article?: string;
  article_title?: string;
  code?: string;
  category?: string;
  severity?: string;
  tier?: number;
  weight?: number;
  confidence?: number;
  message?: string;
  clause_text?: string;
  recommendation?: string;
  blocks_payment?: boolean;
  blocks_payment_reason?: string;
  cascade_impact?: number;
  exclusion_type?: string;
  /** true_violation | contractual_modification | operational_discrepancy */
  finding_nature?: string;
  /** HIGH | MEDIUM | LOW */
  impact_level?: string;
  /** financial | transport | authentication | identity | compliance | document_format */
  clause_type?: string;
  /** Optional extended detail text for the finding */
  detail?: string;
}

export interface RawUcpSummary {
  errors?: number;
  warnings?: number;
  infos?: number;
  exclusions?: number;
  /** Count of findings that block payment */
  blocking?: number;
  /** v4 finding nature breakdown */
  true_violations?: number;
  contractual_modifications?: number;
  operational_discrepancies?: number;
  /** v4 impact level breakdown */
  high_impact?: number;
  medium_impact?: number;
  low_impact?: number;
}

export interface RawUcpCategory {
  score?: number;
  articles_nok?: string[];
  /** Articles that are compliant for this category */
  articles_ok?: string[];
}

export interface RawUcpSemanticMeta {
  mode?: string;
  available?: boolean;
  n_violation?: number;
  n_restriction?: number;
  n_compliant?: number;
}

// ─── DS / External / Risk ML ───────────────────────────────────────────────────

export interface RawDsAnalysisPayload {
  risk?: RawDsRisk;
  features?: Record<string, string | number | boolean>;
  processing_ms?: number;
}

export interface RawDsRisk {
  risk_level?: string;
  risk_score?: number;
  model_used?: string;
  risk_factors?: string[];
}

export interface RawExternalValidationPayload {
  risk_level?: string;
  risk_score?: number;
  risk_factors?: string[];
  recommendations?: string[];
  sources_consulted?: string[];
  processing_ms?: number;
  approach?: string;
  details?: RawExternalDetails;
}

export interface RawExternalDetails {
  countries?: RawExternalCountry[];
  banks?: RawExternalBank[];
  companies?: RawExternalCompany[];
  currency?: RawExternalCurrency | null;
  geo?: Record<string, unknown>;
}

export interface RawExternalCountry {
  raw?: string;
  iso2?: string;
  risk?: string;
  sources?: string[];
  details?: string;
  normalized?: string;
}

export interface RawExternalBank {
  bic?: string;
  gleif_name?: string;
  gleif_status?: string;
  blacklisted?: boolean;
  country?: string;
  risk?: string;
}

export interface RawExternalCompany {
  name?: string;
  risk?: string;
  details?: string;
  ofac_matched?: boolean;
  eu_matched?: boolean;
  un_matched?: boolean;
}

export interface RawExternalCurrency {
  code?: string;
  is_valid?: boolean;
  risk?: string;
  details?: string;
}

export interface RawScoreBreakdownDimension {
  score?: number;
  weight?: number;
  contribution?: number;
  [key: string]: unknown;
}

export interface RawScoreBreakdown {
  analytical_score?: number;
  analytical_label?: string;
  weights?: Record<string, number>;
  dimensions?: {
    completeness?:   RawScoreBreakdownDimension;
    ucp_compliance?: RawScoreBreakdownDimension;
    coherence?:      RawScoreBreakdownDimension;
    country_risk?:   RawScoreBreakdownDimension;
    temporal?:       RawScoreBreakdownDimension;
    amount?:         RawScoreBreakdownDimension;
    ml_model?:       RawScoreBreakdownDimension;
  };
}

export interface RawMissingFieldItem {
  field?: string;
  name?: string;
  criticality?: string;
  impact_score?: number;
}

export interface RawShapWaterfallItem {
  feature?: string;
  value?: number | null;
  shap_value?: number;
  contribution_pct?: number;
  direction?: 'up' | 'down';
  business_direction?: 'reduces_risk' | 'increases_risk' | 'neutral';
  is_discordant?: boolean;
}

export interface RawDiscordantFeature {
  feature?: string;
  value?: number | null;
  shap_value?: number;
  explanation?: string;
}

export interface RawRuleContribution {
  rule?: string;
  field?: string;
  severity?: string;
  description?: string;
  score_impact?: string;
}

export interface RawAnalyticalWaterfallItem {
  dimension:    string;
  label:        string;
  color:        string;
  score:        number;
  weight_pct:   number;
  contribution: number;
  start:        number;
  end:          number;
  risk_tone:    'low' | 'medium' | 'high';
}

export interface RawLimeScoreWaterfallItem {
  feature:            string;
  label:              string;
  value:              number | null;
  shap_value:         number;
  score_contribution: number;  // shap_value × 80 — contribution au risk_score
  start:              number;
  end:                number;
  direction:          'up' | 'down';
  color:              string;
  risk_tone:          'high' | 'low';
}

export interface RawExplanation {
  method?: string;
  final_label?: string;
  risk_score_ml?: number;
  analytical_score?: number;
  // LIME waterfall aligné sur le risk_score (source unique)
  lime_score_waterfall?: RawLimeScoreWaterfallItem[];
  lime_score_base?: number;   // E[P(HIGH)] × 80 = point de départ
  lime_score_total?: number;  // = risk_score
  // Waterfall analytique (référence seulement, non utilisé pour le score final)
  analytical_waterfall?: RawAnalyticalWaterfallItem[];
  // SHAP
  shap_waterfall?: RawShapWaterfallItem[];
  dimension_radar?: Record<string, number>;
  rule_contributions?: RawRuleContribution[];
  top_contributing_dimensions?: { dimension?: string; contribution?: number }[];
  top_shap_features?: RawShapFeature[];
  // Discordances ML vs Métier
  n_discordant_features?: number;
  discordant_features?: RawDiscordantFeature[];
  // Calibration métier v4.1
  _calibration_applied?: boolean;
  _calibration_reason?: string;
  _ml_label_pre_calib?: string;
}

export interface RawUcpDetailViolation {
  article?: string;
  field?: string;
  description?: string;
  severity?: string;
  override_score?: string | null;
}

export interface RawUcpDetail {
  errors_count?: number;
  warnings_count?: number;
  weighted_violation_index?: number;
  error_weight?: number;
  warning_weight?: number;
  compliance_score_pct?: number;
  ucp_decision?: string;
  payment_blocked?: boolean;
  rule_violations_count?: number;
  block_violations?: RawUcpDetailViolation[];
  high_violations?: RawUcpDetailViolation[];
  warn_violations?: RawUcpDetailViolation[];
  errors_detail?: Record<string, unknown>[];
  warnings_detail?: Record<string, unknown>[];
}

export interface RawRiskMlPayload {
  risk_label?: string;
  risk_proba?: Record<string, number>;
  risk_score?: number;
  is_anomaly?: boolean;
  anomaly_score?: number;
  segment?: string;
  segment_id?: number;
  shap_top_features?: RawShapFeature[];
  ucp_violations?: Array<Record<string, unknown>>;
  ucp_detail?: RawUcpDetail;
  narrative?: string;
  features_used?: Record<string, unknown>;
  model_version?: string;
  challenger_label?: string | null;
  processing_ms?: number;
  error?: string | null;
  // v4.0 additions
  score_breakdown?: RawScoreBreakdown;
  missing_fields_detail?: RawMissingFieldItem[];
  explanation?: RawExplanation;
  final_risk_score?: number;
  final_risk_score_components?: Record<string, number>;
}

export interface RawShapFeature {
  feature?: string;
  shap_value?: number;
  direction?: string;
}

export type ValidationPanel =
  | 'overview'
  | 'swift-fields'
  | 'clauses'
  | 'validation'
  | 'ucp'
  | 'coherence'
  | 'external'
  | 'risk-ml'
  | 'perf-cache';
