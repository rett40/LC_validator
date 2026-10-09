export interface LcSummary {
  lcNumber:                string | null;
  beneficiary:             string | null;
  applicant:               string | null;
  currency:                string | null;
  amount:                  number | null;
  tolerancePlus:           number | null;
  toleranceMinus:          number | null;
  expiryDate:              string | null;
  shipmentDate:            string | null;
  presentationPeriodDays:  number | null;
  incoterm:                string | null;
  partialShipmentsAllowed: boolean | null;
  transhipmentAllowed:     boolean | null;
  goodsDescription:        string | null;
  additionalConditions:    string | null;
}

export interface LcCommercialInfo {
  nDocuments:  number;
  docTypes:    string[];
  riskLevel:   string | null;
  crossScore:  number | null;
}

export interface LcRuleResult {
  rule:      string;
  status:    'pass' | 'fail' | 'warning' | 'skipped';
  severity:  'critical' | 'high' | 'medium' | 'low' | 'info';
  message:   string;
  detail:    string;
  lc_value:  string;
  doc_value: string;
  doc_type:  string;
}

export interface LcFuzzyResult {
  check:     string;
  lc_value:  string;
  doc_value: string;
  doc_type:  string;
  score:     number;
  status:    'pass' | 'warning' | 'fail';
  message:   string;
}

export interface LcSemanticResults {
  overall_score:        number;
  lc_vs_invoice?:       number;
  lc_vs_invoice_label?: string;
  lc_vs_bl?:            number;
  lc_vs_bl_label?:      string;
  method?:              string;
  note?:                string;
}

export interface LcCrossExplanation {
  impact:    'positive' | 'negative' | 'warning' | 'neutral';
  severity:  'critical' | 'high' | 'medium' | 'low' | 'info';
  type:      'rule' | 'fuzzy' | 'semantic' | 'condition' | 'category_financial' | 'category_goods' | 'category_documentary' | 'category_shipment' | 'category_parties';
  category:  string;
  message:   string;
}

// v3.0 category-centric interfaces

export interface LcCategoryCheck {
  rule:       string;
  status:     'pass' | 'fail' | 'warning' | 'skipped';
  severity:   'critical' | 'high' | 'medium' | 'low' | 'info';
  message:    string;
  detail?:    string;
  lc_value?:  string;
  doc_value?: string;
  doc_type?:  string;
  weight?:    number;
}

export interface LcCategoryResult {
  score:          number;
  status:         'pass' | 'partial' | 'fail';
  weight:         number;
  primary_issue:  string | null;
  derived_issues: string[];
  checks:         LcCategoryCheck[];
}

export interface LcCrossDocCheck {
  check:      string;
  status:     'pass' | 'warning' | 'fail' | 'skipped';
  severity:   'critical' | 'high' | 'medium' | 'low' | 'info';
  message:    string;
  lc_value?:  string;
  doc_value?: string;
  score?:     number;
}

export interface LcCrossDocPair {
  overall_score: number;
  checks:        LcCrossDocCheck[];
}

export interface LcCrossDocument {
  inv_bl?:       LcCrossDocPair;
  inv_pl?:       LcCrossDocPair;
  bl_pl?:        LcCrossDocPair;
  overall_score: number;
}

// ── XAI Engine output (v3.0 Shapley-based) ───────────────────────────────────

export interface LcWaterfallStep {
  step:          number;
  factor:        string;
  delta:         number;
  cumulative:    number;
  label:         string;
  impact_class:  'baseline' | 'critical_negative' | 'high_negative' | 'low_negative' | 'positive' | 'neutral' | 'final';
  category:      string;
}

export interface LcXaiExplanation {
  rank:           number;
  factor_id:      string;
  shapley_value:  number;
  shapley_pct:    number;
  impact:         'positive' | 'negative' | 'warning';
  severity:       'critical' | 'high' | 'medium' | 'low' | 'info';
  type:           string;
  category:       string;
  rule:           string;
  status:         string;
  message:        string;
  detail:         string;
  remediation:    string;
  lc_value:       string;
  doc_value:      string;
  doc_type:       string;
  is_primary:     boolean;
  ocr_suspect:    boolean;
}

// Level 4 — Simulation step
export interface LcSimulationStep {
  step:           number;
  action:         string;
  score:          number;
  decision:       string;
  gain:           number;
  fix:            string | null;
  category_label: string;
  is_flip:        boolean;
}

// Level 4 — Decision confidence
export interface LcDecisionConfidence {
  probability:            number;
  level:                  string;
  color:                  string;
  margin:                 number;
  closest_competitor:     string | null;
  competitor_probability: number;
  is_borderline:          boolean;
  pct_width:              number;
}

// Level 4A — Probabilistic
export interface LcProbabilistic {
  mean_score:    number;
  std_dev:       number;
  ci_95_lower:   number;
  ci_95_upper:   number;
  decision_proba: { ACCEPT: number; DISCREPANT: number; REJECT: number };
  n_samples:     number;
}

// Level 4B — Causal graph
export interface LcCausalRootCause {
  factor_id:           string;
  rule:                string;
  category:            string;
  shapley_value:       number;
  causal_children:     string[];
  total_causal_impact: number;
  intervention_gain:   number;
  is_root:             boolean;
}
export interface LcCausalChain {
  root:              string;
  chain:             string[];
  total_impact:      number;
  intervention_gain: number;
  n_downstream:      number;
}
export interface LcCausal {
  root_causes:      LcCausalRootCause[];
  causal_chains:    LcCausalChain[];
  derived_failures: string[];
  graph_edges:      string[][];
  n_root_causes:    number;
  n_derived:        number;
}

// Level 4C — Counterfactuals
export interface LcCounterfactual {
  rank:                    number;
  factor_id:               string;
  rule:                    string;
  category:                string;
  category_label:          string;
  rule_label:              string;
  direct_gain:             number;
  causal_gain:             number;
  counterfactual_score:    number;
  counterfactual_decision: string;
  is_root_cause:           boolean;
  n_downstream:            number;
  flips_decision:          boolean;
  message:                 string;
}

export interface LcXaiResult {
  shapley_values:      Record<string, number>;
  waterfall:           LcWaterfallStep[];
  executive_summary:   string;
  decision_rationale:  string;
  compliance_evidence: string[];
  remediation_plan:    string[];
  // Level 4
  probabilistic?:      LcProbabilistic;
  causal?:             LcCausal;
  counterfactuals?:    LcCounterfactual[];
  causal_narrative?:   string;
  simulation?:         LcSimulationStep[];
  decision_confidence?: LcDecisionConfidence;
}

export interface LcCrossValidationResponseDTO {
  swiftAnalysisId:        number;
  commercialValidationId: number;
  finalScore:             number;
  riskLevel:              'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | string;
  /** LC-Commercial cross-validation decision */
  decision?:              'ACCEPT' | 'DISCREPANT' | 'REJECT' | string;
  /** UCP 600 compliance decision (5-level) */
  ucpDecision?:           'ACCEPT' | 'ACCEPT_WITH_FLAGS' | 'REVIEW' | 'CONDITIONAL_HOLD' | 'REJECT' | string;
  contractualHold?:       boolean;

  scoreBreakdown:   Record<string, number>;
  weights:          Record<string, number>;
  categoryWeights?: Record<string, number>;

  categories?:    Record<string, LcCategoryResult>;
  crossDocument?: LcCrossDocument;
  xai?:           LcXaiResult;

  ruleResults:      LcRuleResult[];
  fuzzyResults:     LcFuzzyResult[];
  semanticResults:  LcSemanticResults;
  conditionResults: LcRuleResult[];
  explanations:     LcCrossExplanation[];
  lcSummary:        LcSummary;
  commercialInfo:   LcCommercialInfo;
}
