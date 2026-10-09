export interface CommercialRuleSummary {
  pass: number;
  fail: number;
  warning: number;
  corrected: number;
}

export interface CommercialDocumentResult {
  id: number;
  sourceFilename: string;
  status: 'success' | 'error';
  docType: 'bl' | 'invoice' | 'packing_list' | string;
  inputType: 'pdf_native' | 'pdf_scanned' | 'image' | string;
  confidence: number;
  confident: boolean;
  extractionScore: number;
  nFields: number;
  nPages: number;
  nWarnings: number;
  nRuleFails: number;
  fields: Record<string, string>;
  tables: Record<string, unknown[]>;
  extractionTiers: Record<string, string>;
  ruleResults: CommercialRuleResult[];
  ruleSummary: CommercialRuleSummary;
  processingTimeS: number;
  warnings: string[];
  error?: string;
}

export interface CommercialRuleResult {
  rule: string;
  status: 'pass' | 'fail' | 'warning' | 'corrected' | 'skipped';
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  message: string;
  field?: string;
  doc_pair?: string;
  original?: string;
}

export interface CrossSemanticScore {
  [key: string]: number;
}

export interface CrossNliResult {
  nli_label: 'ENTAILMENT' | 'CONTRADICTION' | 'NEUTRAL';
  score: number;
  description: string;
  category: string;
  premise: string;
  hypothesis: string;
  all_scores: Record<string, number>;
}

export interface CrossTableMatch {
  doc_pair: string;
  invoice_rows?: number;
  pl_rows?: number;
  matched_rows?: number;
  match_rate?: number;
  unmatched_rows?: unknown[];
  bl_containers?: string[];
  pl_containers?: string[];
  common_containers?: string[];
  alignment_rate?: number;
}

export interface CrossExplanation {
  impact: 'positive' | 'negative' | 'warning' | 'neutral';
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  type: string;
  category: string;
  message: string;
}

export interface CrossScoreBreakdown {
  semantic?: number;
  nli?: number;
  rules?: number;
  table?: number;
  party?: number;
}

export interface CrossReportDTO {
  finalScore: number | null;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | string | null;
  scoreBreakdown: CrossScoreBreakdown;
  ruleResults: CommercialRuleResult[];
  semanticScores: CrossSemanticScore;
  nliResults: CrossNliResult[];
  tableMatches: CrossTableMatch[];
  explanations: CrossExplanation[];
  documents: unknown[];
}

export interface CommercialSummaryDTO {
  totalDocuments: number;
  processed: number;
  failed: number;
  processingTimeS: number;
}

export interface CommercialValidationResponseDTO {
  id: number;
  lcReference: string | null;
  swiftAnalysisId: number | null;
  status: string;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | string | null;
  finalScore: number | null;
  nDocuments: number;
  nProcessed: number;
  nFailed: number;
  processingTimeS: number;
  createdAt: string;

  summary: CommercialSummaryDTO;
  documents: CommercialDocumentResult[];
  crossValidation: CommercialRuleResult[];
  crossReport: CrossReportDTO | null;
  fullPayloadJson?: string | null;
}
