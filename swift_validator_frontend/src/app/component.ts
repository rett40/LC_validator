import { Component, OnInit, signal, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ReportService, ReportMeta } from './core/services/report.service';
import { AnalysisService, PageResponse } from './core/services/analysis.service';
import {
  AnalysisResponseDTO,
  DashboardStatsDTO,
  RawAnalysisPayload,
  RawClause,
  RawContradiction,
  RawCorrection,
  RawExternalCompany,
  RawExternalCountry,
  RawExternalCurrency,
  RawGeoEntry,
  RawRiskMlPayload,
  RawShapFeature,
  RawSwiftEntities,
  RawValidationField,
  ValidationPanel
} from './core/models/analysis.model';
import { AuthService } from './core/services/auth.service';
import { RiskBreakdownComponent } from './features/risk-breakdown/risk-breakdown.component';

interface PanelConfig {
  id: ValidationPanel;
  icon: string;
  label: string;
  count?: number;
}

interface SwiftFieldRow {
  tag: string;
  name: string;
  value: string;
  status: 'ok' | 'warn' | 'error' | 'neutral';
  mandatory: boolean;
  sourceBadges: string[];
  confidence: number | null;
  errors: string[];
  warnings: string[];
}

interface UnifiedContradiction {
  id: string;
  source: 'NLP' | 'SEMANTIC';
  severity: string;
  message: string;
  clauseA: string;
  clauseB: string;
  ucpRef?: string;
  confidence?: number;
}

interface GeoRow {
  key: string;
  status: string;
  country: string;
  location: string;
  risk: string;
  details: string;
}

interface TimingBar {
  key: string;
  label: string;
  ms: number;
  tone: string;
}

interface HeroBadge {
  label: string;
  tone: string;
}

interface KpiCard {
  icon: string;
  label: string;
  value: string;
  tone: string;
}

interface ScorePill {
  label: string;
  value: string;
  tone: string;
}

interface SummaryFact {
  label: string;
  value: string;
  tone?: string;
}

interface FeatureRow {
  key: string;
  value: string;
}

interface ScoreComponent {
  label: string;
  weight: number;
  rawScore: number;
  contribution: number;
  color: string;
}

interface CrossFieldGroup {
  axis: string;
  label: string;
  issues: any[];
  errorCount: number;
  warningCount: number;
}

// FIX: Removed 'additional_conditions' (tag 47A) from blueprint to hide it from SWIFT fields table
const SWIFT_FIELD_BLUEPRINT: Array<{ tag: string; key: string; label: string }> = [
  { tag: '27',  key: 'sequence',            label: 'Sequence' },
  { tag: '40A', key: 'lc_form',             label: 'Forme du credit' },
  { tag: '20',  key: 'lc_number',           label: 'Numero LC' },
  { tag: '31C', key: 'issue_date',          label: 'Date emission' },
  { tag: '40E', key: 'applicable_rules',    label: 'Regles applicables' },
  { tag: '31D', key: 'expiry_date',         label: 'Date expiration' },
  { tag: '50',  key: 'applicant',           label: 'Applicant' },
  { tag: '59',  key: 'beneficiary',         label: 'Beneficiary' },
  { tag: '52A', key: 'issuing_bank',        label: 'Banque emettrice' },
  { tag: '32B', key: 'amount',              label: 'Montant' },
  { tag: '39A', key: 'tolerance',           label: 'Tolerance' },
  { tag: '41D', key: 'available_with',      label: 'Disponible avec' },
  { tag: '41D-BY', key: 'available_by',     label: 'Disponible par' },
  { tag: '42C', key: 'draft_at',            label: 'Draft at' },
  { tag: '43P', key: 'partial_shipments',   label: 'Partial shipments' },
  { tag: '43T', key: 'transhipment',        label: 'Transhipment' },
  { tag: '44E', key: 'port_loading',        label: 'Port of loading' },
  { tag: '44F', key: 'port_discharge',      label: 'Port of discharge' },
  { tag: '44C', key: 'shipment_date',       label: 'Date expedition' },
  { tag: '48',  key: 'presentation_period', label: 'Presentation period' },
  { tag: '49',  key: 'confirmation',        label: 'Confirmation' },
  { tag: '45A', key: 'goods_description',   label: 'Marchandises' },
  { tag: '46A', key: 'documents_required',  label: 'Documents requis' },
  // REMOVED: { tag: '47A', key: 'additional_conditions', label: 'Conditions additionnelles' },
  { tag: '71B', key: 'charges',             label: 'Charges' },
  { tag: '72',  key: 'sender_info',         label: 'Info emetteur' },
  { tag: '77U', key: 'guarantee_text',      label: 'Texte garantie' }
];

// Tags to always exclude from the SWIFT fields table
const EXCLUDED_SWIFT_TAGS = new Set(['47A', 'ADDLCOND', 'ADDITIONAL_CONDITIONS']);

const CLAUSE_TYPE_META: Record<string, { icon: string; label: string }> = {
  document_condition:  { icon: '📄', label: 'Document' },
  shipment_condition:  { icon: '🚢', label: 'Expedition' },
  payment_condition:   { icon: '💰', label: 'Paiement' },
  tolerance_condition: { icon: '📏', label: 'Tolerance' },
  compliance_condition:{ icon: '⚖️', label: 'Compliance' }
};

const NER_LABELS_BY_FIELD: Record<string, string[]> = {
  applicant:         ['APPLICANT'],
  beneficiary:       ['BENEFICIARY'],
  issuing_bank:      ['ISSUING_BANK', 'BANK'],
  amount:            ['AMOUNT'],
  currency:          ['CURRENCY'],
  issue_date:        ['ISSUE_DATE'],
  expiry_date:       ['EXPIRY_DATE'],
  shipment_date:     ['SHIPMENT_DATE'],
  goods_description: ['GOODS', 'GOODS_DESCRIPTION'],
  incoterm:          ['INCOTERM'],
  country:           ['COUNTRY']
};

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './component.v2.html',
  styleUrls: ['./component.v2.scss'],
  imports: [CommonModule, FormsModule, RiskBreakdownComponent]
})
export class AppComponent implements OnInit {
  readonly maxHistoryItems = 20;

  isDragging    = false;
  selectedFile: File | null = null;
  uploading     = false;
  uploadError   = '';
  uiState: 'idle' | 'uploading' | 'success' | 'error' = 'idle';
  detailLoading = false;

  analysis:        AnalysisResponseDTO | null = null;
  analysisPayload: RawAnalysisPayload  | null = null;
  stats:           DashboardStatsDTO   | null = null;
  history         = signal<AnalysisResponseDTO[]>([]);

  activePanel:  ValidationPanel = 'overview';
  showHistory   = false;
  historyLoading = signal(false);

  validationPanels:   PanelConfig[]         = [];
  clauseTypes:        string[]              = [];
  swiftFieldRows:     SwiftFieldRow[]       = [];
  unifiedContradictions: UnifiedContradiction[] = [];
  geoRows:            GeoRow[]              = [];
  perfBars:           TimingBar[]           = [];
  shapMaxAbs          = 0;

  clauseFilterType:    string                         = 'ALL';
  clauseFilterRisk:    string                         = 'ALL';
  contradictionFilter: 'ALL' | 'NLP' | 'SEMANTIC'    = 'ALL';
  ucpFindingFilter:    'ALL' | 'error' | 'warning' | 'info' = 'ALL';

  // ── Rapports PDF ───────────────────────────────────────────────────
  /** Rapports de l'analyse actuellement affichée (espace validation). */
  currentReports: ReportMeta[] = [];
  reportBusy = false;
  reportError = '';

  /** Charge les rapports déjà générés pour une analyse donnée. */
  loadReportsFor(analysisId: number): void {
    this.reportService.listReports(analysisId).subscribe({
      next: (reports) => { this.currentReports = reports ?? []; this.cdr.markForCheck(); },
      error: () => { this.currentReports = []; },
    });
  }

  /** Génère un rapport pour l'analyse courante (espace validation). */
  generateCurrentReport(type: 'MT700_REPORT' | 'FULL_REPORT'): void {
    if (!this.analysis) { return; }
    this.reportBusy = true;
    this.reportError = '';
    const id = this.analysis.id;
    const obs = type === 'MT700_REPORT'
      ? this.reportService.generateMt700(id)
      : this.reportService.generateFull(id);
    obs.subscribe({
      next: (report) => {
        // remplace un éventuel rapport du même type, puis ouvre le PDF
        this.currentReports = [report, ...this.currentReports.filter(r => r.reportType !== report.reportType)];
        this.reportBusy = false;
        this.reportService.openPdf(report);
        this.cdr.markForCheck();
      },
      error: () => {
        this.reportBusy = false;
        this.reportError = 'Échec de génération du rapport. Réessayez.';
        this.cdr.markForCheck();
      },
    });
  }

  openReport(r: ReportMeta): void {
    this.reportService.openPdf(r);
  }

  get mt700Report(): ReportMeta | null {
    return this.currentReports.find(r => r.reportType === 'MT700_REPORT') ?? null;
  }

  /** Génération depuis une carte d'historique. */
  generateReport(item: AnalysisResponseDTO & { generatingReport?: boolean; reports?: ReportMeta[] },
                 type: 'MT700_REPORT' | 'FULL_REPORT'): void {
    item.generatingReport = true;
    const obs = type === 'MT700_REPORT'
      ? this.reportService.generateMt700(item.id)
      : this.reportService.generateFull(item.id);
    obs.subscribe({
      next: (report) => {
        if (!item.reports) item.reports = [];
        item.reports = [report, ...item.reports.filter(r => r.reportType !== report.reportType)];
        item.generatingReport = false;
        this.reportService.openPdf(report);
        this.cdr.markForCheck();
      },
      error: () => { item.generatingReport = false; },
    });
  }

  constructor(
    private readonly analysisService: AnalysisService,
    private readonly auth: AuthService,
    private readonly router: Router,
    private readonly cdr: ChangeDetectorRef,
    private readonly reportService: ReportService
  ) {}

  ngOnInit(): void {
    this.validationPanels = this.basePanels();
    this.loadStats();
    this.loadHistory();
  }

  loadStats(): void {
    this.analysisService.getDashboardStats().subscribe({
      next: (res: DashboardStatsDTO) => { this.stats = res; },
      error: () => { this.stats = null; }
    });
  }

  loadHistory(): void {
    this.historyLoading.set(true);

    const safetyTimer = setTimeout(() => {
      if (this.historyLoading()) {
        this.historyLoading.set(false);
      }
    }, 15000);

    this.analysisService.list().subscribe({
      next: (res: PageResponse<AnalysisResponseDTO>) => {
        clearTimeout(safetyTimer);
        this.history.set(res?.content ?? []);
        this.historyLoading.set(false);
      },
      error: () => {
        clearTimeout(safetyTimer);
        this.history.set([]);
        this.historyLoading.set(false);
      }
    });
  }

  loadFromHistory(item: AnalysisResponseDTO): void {
    this.showHistory   = false;
    this.uploadError   = '';
    this.uiState       = 'success';
    this.activePanel   = 'overview';

    this.analysis = item;
    this.syncDerivedData();
    this.loadReportsFor(item.id);

    if (item.rawPayloadJson) {
      return;
    }

    this.detailLoading = true;
    this.analysisService.getById(item.id).subscribe({
      next: (res: AnalysisResponseDTO) => {
        this.analysis = res;
        this.syncDerivedData();
        this.detailLoading = false;
      },
      error: () => {
        this.detailLoading = false;
      }
    });
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = true;
  }

  onDragLeave(): void {
    this.isDragging = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = false;
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.selectedFile = files[0];
    }
  }

  onFileChange(event: Event): void {
    const input = event.target as HTMLInputElement | null;
    const file  = input?.files?.[0] ?? null;
    if (file) {
      this.selectedFile = file;
    }
  }

  analyze(): void {
    if (!this.selectedFile || this.uploading) {
      return;
    }

    this.showHistory   = false;
    this.activePanel   = 'overview';
    this.uploading     = true;
    this.uiState       = 'uploading';
    this.uploadError   = '';
    this.detailLoading = false;

    this.analysisService.upload(this.selectedFile).subscribe({
      next: (res: AnalysisResponseDTO) => {
        this.analysis = res;
        this.syncDerivedData();
        this.uploading    = false;
        this.selectedFile = null;
        this.uiState      = 'success';
        this.currentReports = [];
        this.loadReportsFor(res.id);
        this.cdr.detectChanges();
        this.loadStats();
        this.loadHistory();
      },
      error: (err: HttpErrorResponse) => {
        if (err.status === 401 || err.status === 403) {
          this.uploadError = 'Session expiree, veuillez vous reconnecter.';
          this.router.navigateByUrl('/login');
        } else if (err.status === 0) {
          this.uploadError = 'Backend indisponible. Verifiez les services Spring et Python.';
        } else if (err.status === 408 || err.status === 504) {
          this.uploadError = 'Analyse trop longue. Essayez un document plus leger.';
        } else {
          this.uploadError = err.error?.message || "Erreur lors de l'analyse.";
        }
        this.uploading = false;
        this.uiState   = 'error';
      }
    });
  }

  navigateToCommercial(): void {
    const id = this.analysis?.id;
    if (id != null) {
      this.router.navigate(['/commercial'], { queryParams: { swiftAnalysisId: id } });
    } else {
      this.router.navigateByUrl('/commercial');
    }
  }

  logout(): void {
    this.auth.logout().subscribe({
      next:  () => { this.router.navigateByUrl('/login'); },
      error: () => { this.router.navigateByUrl('/login'); }
    });
  }

  setPanel(panel: ValidationPanel): void {
    this.activePanel = panel;
  }

  formatDate(date: string | Date | null | undefined): string {
    if (!date) { return '—'; }
    return new Date(date).toLocaleDateString('fr-FR');
  }

  formatAmount(amount: number | null | undefined, currency = ''): string {
    if (amount == null || Number.isNaN(amount)) { return '—'; }
    const safeCurrency = currency && currency.length === 3 ? currency : '';
    return new Intl.NumberFormat('fr-FR', {
      style:                safeCurrency ? 'currency' : 'decimal',
      currency:             safeCurrency || 'EUR',
      maximumFractionDigits: safeCurrency ? 2 : 0
    }).format(amount);
  }

  scoreColor(score: number | null | undefined): string {
    const value = score ?? 0;
    if (value >= 80) { return 'green'; }
    if (value >= 60) { return 'orange'; }
    return 'red';
  }

  /** Code couleur INVERSÉ pour un score de risque (élevé = rouge). */
  riskScoreColor(score: number | null | undefined): string {
    const value = score ?? 0;
    if (value >= 66) { return 'red'; }
    if (value >= 33) { return 'orange'; }
    return 'green';
  }

  riskColor(level: string | null | undefined): string {
    const tone = (level || '').toUpperCase();
    if (tone === 'CLEAN' || tone === 'LOW')             { return 'green'; }
    if (tone === 'MEDIUM')                              { return 'orange'; }
    if (tone === 'HIGH' || tone === 'CRITICAL')         { return 'red'; }
    return 'gray';
  }

  clauseRiskClass(level: string | undefined): string {
    const risk = (level || '').toLowerCase();
    if (risk === 'high')                                       { return 'risk-high'; }
    if (risk === 'medium')                                     { return 'risk-medium'; }
    if (risk === 'low' || risk === 'normal' || risk === 'clean') { return 'risk-low'; }
    return 'risk-neutral';
  }

  /** Couleur de la pill d'importance de clause (signal unique conservé). */
  importanceClass(importance: string | undefined): string {
    const v = (importance || '').toLowerCase();
    if (v === 'high')   { return 'red'; }
    if (v === 'medium') { return 'orange'; }
    return 'soft';
  }

  /** Libellé lisible de l'importance. */
  importanceLabel(importance: string | undefined): string {
    const v = (importance || 'normal').toLowerCase();
    return { high: 'Élevée', medium: 'Moyenne', normal: 'Normale' }[v] ?? v.toUpperCase();
  }

  severityClass(value: string | undefined): string {
    const s = (value || '').toLowerCase();
    if (s.includes('error') || s.includes('critical') || s.includes('high')) { return 'red'; }
    if (s.includes('warning') || s.includes('medium'))                        { return 'orange'; }
    if (s.includes('ok') || s.includes('info') || s.includes('low'))          { return 'green'; }
    return 'gray';
  }

  statusDot(status: SwiftFieldRow['status']): string {
    if (status === 'ok')    { return '🟢'; }
    if (status === 'warn')  { return '🟡'; }
    if (status === 'error') { return '🔴'; }
    return '⚪';
  }

  clauseTypeLabel(type: string | undefined): string {
    const safeType = type || 'unknown';
    return CLAUSE_TYPE_META[safeType]?.label || this.humanize(safeType);
  }

  clauseTypeIcon(type: string | undefined): string {
    const safeType = type || 'unknown';
    return CLAUSE_TYPE_META[safeType]?.icon || '📌';
  }

  clauseMethodLabel(method: string | undefined): string {
    const value = (method || '').toLowerCase();
    return value.includes('embedding') || value.includes('supervised') ? 'ML' : 'Regle';
  }

  shapBarWidth(value: number | undefined): number {
    const current = Math.abs(value ?? 0);
    if (!this.shapMaxAbs) { return 0; }
    return Math.round((current / this.shapMaxAbs) * 100);
  }

  shapSignClass(value: number | undefined): string {
    if ((value ?? 0) > 0) { return 'positive'; }
    if ((value ?? 0) < 0) { return 'negative'; }
    return 'neutral';
  }

  renderPercent(value: number | null | undefined, scale = 100): string {
    if (value == null) { return '—'; }
    const normalized = scale === 1 ? Math.round(value * 100) : Math.round(value);
    return `${normalized}%`;
  }

  renderConfidence(value: number | null | undefined): string {
    if (value == null) { return '—'; }
    return `${Math.round(value * 100)}%`;
  }

  normalizeValue(value: unknown): string {
    if (value == null)             { return '—'; }
    if (typeof value === 'string') { return value.trim() || '—'; }
    if (typeof value === 'number') { return Number.isFinite(value) ? String(value) : '—'; }
    if (typeof value === 'boolean'){ return value ? 'Oui' : 'Non'; }
    try   { return JSON.stringify(value); }
    catch { return String(value); }
  }

  truncate(text: string | null | undefined, limit = 350): string {
    const value = (text || '').trim();
    if (!value) { return '—'; }
    return value.length > limit ? `${value.slice(0, limit)}…` : value;
  }

  // ── Convenience getters ────────────────────────────────────

  get hasAnalysis(): boolean { return !!this.analysis; }

  get rawSwift()      { return this.analysisPayload?.swift ?? null; }
  get rawValidation() { return this.rawSwift?.validation ?? null; }
  get rawNlp()        { return this.rawSwift?.nlp ?? null; }
  get rawSemantic()   { return this.rawNlp?.semantic ?? null; }
  get rawNer()        { return this.rawNlp?.ner ?? null; }
  get rawUcp()        { return this.analysisPayload?.ucp_compliance ?? null; }
  get rawFinalDecision() { return this.analysisPayload?.final_decision ?? null; }
  get rawDs()         { return this.analysisPayload?.ds_analysis ?? null; }
  get rawExternal()   { return this.analysisPayload?.external_validation ?? null; }
  get rawRiskMl(): RawRiskMlPayload | null { return this.analysisPayload?.risk_ml ?? null; }

  /** Score de conformité documentaire 0-100 (qualité du document : UCP + NLP + Validation + Complétude). */
  get globalScore(): number {
    const validationScore = this.rawValidation?.score ?? this.analysis?.swiftValidation?.score;
    const ucpScore        = this.rawUcp?.global_score ?? this.analysis?.ucpComplianceScore;
    const nlpScore        = this.rawNlp?.score ?? this.analysis?.nlpScore;
    const completeness    = this.rawSemantic?.completeness_score ?? this.analysis?.completenessScore;
    const scores = [validationScore, ucpScore, nlpScore, completeness]
      .filter((v): v is number => typeof v === 'number' && v >= 0);
    if (!scores.length) { return 0; }
    return Math.round(scores.reduce((t, v) => t + v, 0) / scores.length);
  }

  /**
   * Score de risque réglementaire 0-100 — source canonique unifiée.
   * Priorité : finalRiskScore (backend normalisé) > ML final_risk_score > analytical_score > DS score normalisé.
   * Distinct du globalScore (conformité) : un document peut être CONFORME (globalScore élevé)
   * mais rester à risque MOYEN pour des raisons réglementaires (pays, tenor, etc.).
   */
  get primaryRiskScore(): number {
    // 1. finalRiskScore normalisé depuis le backend (0-100)
    if (typeof this.analysis?.finalRiskScore === 'number' && this.analysis.finalRiskScore >= 0) {
      return this.analysis.finalRiskScore;
    }
    // 2. ML final_risk_score (0-100)
    const mlFinal = this.rawRiskMl?.final_risk_score;
    if (typeof mlFinal === 'number' && mlFinal >= 0) return mlFinal;
    // 3. Analytical score du breakdown (0-100)
    const analytical = this.rawRiskMl?.score_breakdown?.analytical_score;
    if (typeof analytical === 'number' && analytical >= 0) return Math.round(analytical);
    // 4. ML risk_score (0-100)
    const mlScore = this.rawRiskMl?.risk_score;
    if (typeof mlScore === 'number' && mlScore >= 0) return mlScore;
    // 5. DS riskScore normalisé (0-1 → 0-100)
    const ds = this.analysis?.riskScore;
    if (typeof ds === 'number' && ds >= 0) {
      return Math.round(ds <= 1.0 ? ds * 100 : ds);
    }
    return 0;
  }

  get isCalibrated(): boolean {
    return !!(this.rawRiskMl?.explanation?.['_calibration_applied']);
  }
  get calibrationReason(): string {
    return String(this.rawRiskMl?.explanation?.['_calibration_reason'] ?? '');
  }

  /**
   * Verdict hero cohérent : ancré sur la décision réconciliée (Phase 1, qui intègre
   * les hard-blocks UCP) ET escaladé par le risque ML. 3 niveaux. Plus de « CONFORME »
   * quand le risque est élevé / des erreurs existent, même si aucun override ne déclenche.
   */
  get heroVerdictKind(): 'compliant' | 'expert' | 'noncompliant' {
    const risk = this.primaryRiskScore;
    const dec  = this.finalDecision;
    if (dec === 'REJECT' || risk >= 66) { return 'noncompliant'; }
    if (dec === 'REVIEW' || dec === 'CONDITIONAL_HOLD' || risk >= 33) { return 'expert'; }
    return 'compliant';
  }

  get verdictLabel(): string {
    return {
      compliant:    'CONFORME',
      expert:       'NÉCESSITE UN EXPERT',
      noncompliant: 'NON CONFORME',
    }[this.heroVerdictKind];
  }

  get verdictTone(): string {
    return {
      compliant:    'green',
      expert:       'orange',
      noncompliant: 'red',
    }[this.heroVerdictKind];
  }

  get verdictDescription(): string {
    const missing = this.rawValidation?.missing_mandatory?.length ?? 0;
    const flags   = this.rawSemantic?.risk_flags?.length ?? 0;
    const risk    = this.primaryRiskScore;
    switch (this.heroVerdictKind) {
      case 'compliant':
        return 'Analyse coherente, risque faible et champs obligatoires couverts.';
      case 'noncompliant':
        if (this.finalDecision === 'REJECT') {
          return 'Rejet : violation bloquante ou risque eleve (voir score de risque et UCP).';
        }
        return missing
          ? `${missing} champ(s) obligatoire(s) manquant(s) — risque ${risk}/100.`
          : `Risque eleve (${risk}/100) — non conforme.`;
      default: // expert
        return flags
          ? `Risque modere (${risk}/100) et ${flags} risk flag(s) — revue par un expert recommandee.`
          : `Risque modere (${risk}/100) — revue par un expert recommandee.`;
    }
  }

  get heroBadges(): HeroBadge[] {
    const badges: HeroBadge[] = [];
    if (this.rawNlp?.from_cache)    { badges.push({ label: '⚡ CACHE',   tone: 'blue' }); }
    if (this.rawNlp?.llm_triggered) { badges.push({ label: '🤖 LLM',    tone: 'purple' }); }
    const sourceMode = (this.rawNer?.source_mode || '').toLowerCase();
    if (sourceMode === 'rules')     { badges.push({ label: 'RULES',      tone: 'gray' }); }
    else if (sourceMode)            { badges.push({ label: 'NER_HIGH_CONF', tone: 'green' }); }
    return badges;
  }

  get globalKpis(): KpiCard[] {
    const validated = this.swiftFieldRows.filter(r => r.status === 'ok').length;
    return [
      { icon: '🔑', label: 'Entités extraites', value: String(this.rawNer?.entities?.length ?? 0), tone: 'blue' },
      { icon: '📌', label: 'Clauses',           value: String(this.rawNlp?.clauses?.length ?? 0),  tone: 'blue' },
      { icon: '⚠️', label: 'Contradictions',    value: String(this.meaningfulContradictions.length),
        tone: this.meaningfulContradictions.length ? 'orange' : 'green' },
      { icon: '🔴', label: 'Champs manquants',  value: String(this.rawValidation?.missing_mandatory?.length ?? 0),
        tone: (this.rawValidation?.missing_mandatory?.length ?? 0) ? 'red' : 'green' },
      { icon: '🚩', label: 'Risk flags',        value: String(this.rawSemantic?.risk_flags?.length ?? 0),
        tone: (this.rawSemantic?.risk_flags?.length ?? 0) ? 'red' : 'green' },
      { icon: '📄', label: 'Doc Quality',       value: this.renderPercent(this.rawSemantic?.doc_quality_score),
        tone: this.scoreColor(this.rawSemantic?.doc_quality_score) },
      { icon: '📋', label: 'Champs valides',    value: String(validated),
        tone: validated ? 'green' : 'gray' }
    ];
  }

  get detailedScorePills(): ScorePill[] {
    const ocrConfidence = this.analysisPayload?.ocr?.confidence;
    return [
      { label: 'Analyse documentaire', value: `${this.rawNlp?.score ?? this.analysis?.nlpScore ?? 0}/100`,
        tone: this.scoreColor(this.rawNlp?.score ?? this.analysis?.nlpScore) },
      { label: 'Validation', value: `${this.rawValidation?.score ?? this.analysis?.swiftValidation?.score ?? 0}/100`,
        tone: this.scoreColor(this.rawValidation?.score ?? this.analysis?.swiftValidation?.score) },
      { label: 'UCP',        value: `${this.rawUcp?.global_score ?? this.analysis?.ucpComplianceScore ?? 0}/100`,
        tone: this.scoreColor(this.rawUcp?.global_score ?? this.analysis?.ucpComplianceScore) },
      { label: 'Completude', value: this.renderPercent(this.rawSemantic?.completeness_score ?? this.analysis?.completenessScore),
        tone: this.scoreColor(this.rawSemantic?.completeness_score ?? this.analysis?.completenessScore) },
      { label: 'Doc Quality',value: this.renderPercent(this.rawSemantic?.doc_quality_score),
        tone: this.scoreColor(this.rawSemantic?.doc_quality_score) },
      { label: 'Ext',        value: this.rawExternal?.risk_level || this.analysis?.externalValidation?.riskLevel || '—',
        tone: this.riskColor(this.rawExternal?.risk_level || this.analysis?.externalValidation?.riskLevel) },
      { label: 'DS',         value: this.rawDs?.risk?.risk_level || this.analysis?.dsRiskScore?.riskLevel || '—',
        tone: this.riskColor(this.rawDs?.risk?.risk_level || this.analysis?.dsRiskScore?.riskLevel) },
      { label: 'ML',         value: this.rawRiskMl?.error ? 'Indispo' : this.rawRiskMl?.risk_label || this.analysis?.riskMl?.riskLabel || '—',
        tone: this.riskColor(this.rawRiskMl?.risk_label || this.analysis?.riskMl?.riskLabel) },
      { label: 'OCR',        value: this.renderPercent(ocrConfidence, 1),
        tone: this.scoreColor(ocrConfidence != null ? ocrConfidence * 100 : null) }
    ];
  }

  get overviewPartyFacts(): SummaryFact[] {
    return [
      { label: 'Applicant',      value: this.analysis?.applicant || this.normalizeValue(this.rawSwift?.entities?.applicant) },
      { label: 'Beneficiary',    value: this.analysis?.beneficiary || this.normalizeValue(this.rawSwift?.entities?.beneficiary) },
      { label: 'Banque emettrice',value: this.analysis?.issuingBank || this.normalizeValue(this.rawSwift?.entities?.issuing_bank) }
    ];
  }

  get overviewTradeFacts(): SummaryFact[] {
    return [
      { label: 'Montant',        value: this.formatAmount(this.analysis?.amount, this.analysis?.currency), tone: 'blue' },
      { label: 'Incoterm',       value: this.analysis?.incoterm || this.normalizeValue(this.rawSwift?.entities?.incoterm) },
      { label: 'Date emission',  value: this.formatDate(this.analysis?.issueDate || this.normalizeValue(this.rawSwift?.entities?.issue_date)) },
      { label: 'Date expiration',value: this.formatDate(this.analysis?.expiryDate || this.normalizeValue(this.rawSwift?.entities?.expiry_date)) },
      { label: 'Date expedition',value: this.formatDate(this.analysis?.shipmentDate || this.normalizeValue(this.rawSwift?.entities?.shipment_date)) },
      { label: 'Presentation',   value: this.analysis?.presentationPeriodDays != null
          ? `${this.analysis.presentationPeriodDays} jours`
          : this.normalizeValue(this.rawSwift?.entities?.presentation_period) }
    ];
  }

  get overviewNarratives(): Array<{ title: string; content: string }> {
    return [
      { title: 'Marchandises',          content: this.analysis?.goodsDescription || this.normalizeValue(this.rawSwift?.entities?.goods_description) },
      { title: 'Documents requis',      content: this.analysis?.documentsRequired || this.normalizeValue(this.rawSwift?.entities?.documents_required) },
      { title: 'Conditions additionnelles', content: this.analysis?.additionalConditions || this.normalizeValue(this.rawSwift?.entities?.additional_conditions) }
    ].filter(item => item.content !== '—');
  }

  get validationErrorFields():   SwiftFieldRow[] { return this.swiftFieldRows.filter(r => r.status === 'error'); }
  get validationWarningFields():  SwiftFieldRow[] { return this.swiftFieldRows.filter(r => r.status === 'warn'); }
  get validationOkFields():       SwiftFieldRow[] { return this.swiftFieldRows.filter(r => r.status === 'ok'); }
  get validationScoreWidth():     number          { return this.rawValidation?.score ?? this.analysis?.swiftValidation?.score ?? 0; }

  get ucpErrorCount():   number { return this.rawUcp?.findings?.filter(i => i.severity === 'error').length   ?? 0; }
  get ucpWarningCount(): number { return this.rawUcp?.findings?.filter(i => i.severity === 'warning').length ?? 0; }
  get ucpInfoCount():    number { return this.rawUcp?.findings?.filter(i => i.severity === 'info').length    ?? 0; }

  // ── Décision UCP : rendre le « pourquoi » lisible ─────────────────────
  /** Décision normalisée (le back peut renvoyer une string ou { decision }). */
  get ucpDecision(): string {
    const d = this.rawUcp?.ucp_decision;
    const raw = typeof d === 'object' && d ? (d.decision ?? '') : (d ?? '');
    if (raw) { return String(raw).toUpperCase(); }
    // Fallback si la décision n'est pas fournie : on déduit depuis compliant/payment_blocked
    if (this.rawUcp?.payment_blocked) { return 'REJECT'; }
    return this.rawUcp?.compliant ? 'ACCEPT' : 'REVIEW';
  }

  /** Libellé court affiché dans le badge. */
  get ucpDecisionLabel(): string {
    const labels: Record<string, string> = {
      ACCEPT:            '✓ ACCEPTÉ',
      ACCEPT_WITH_FLAGS: '✓ ACCEPTÉ (alertes)',
      REVIEW:            '🟡 REVUE REQUISE',
      CONDITIONAL_HOLD:  '🟠 SOUS RÉSERVE',
      REJECT:            '✕ REJETÉ',
    };
    return labels[this.ucpDecision] ?? this.ucpDecision;
  }

  /** Phrase explicative : POURQUOI cette décision. */
  get ucpDecisionExplanation(): string {
    const n = this.rawUcp?.ucp_blocking_violations?.length ?? 0;
    const explanations: Record<string, string> = {
      ACCEPT:            'Aucune anomalie UCP 600 détectée.',
      ACCEPT_WITH_FLAGS: 'Conforme : seulement des alertes mineures ou dérogations contractuelles non bloquantes.',
      REVIEW:            'Erreur(s) UCP sans blocage direct du paiement — vérification manuelle requise.',
      CONDITIONAL_HOLD:  'Dérogation contractuelle (Art.1 UCP) à risque structurel élevé — paiement sous réserve, pas une violation UCP.',
      REJECT:            n
        ? `Rejeté : ${n} violation${n > 1 ? 's' : ''} UCP réelle${n > 1 ? 's' : ''} bloque${n > 1 ? 'nt' : ''} le paiement (voir ci-dessous).`
        : 'Rejeté : au moins une violation UCP réelle bloque le paiement.',
    };
    return explanations[this.ucpDecision] ?? '';
  }

  /** Classe CSS du badge (PASS / REVIEW / REJECT) — ne réduit plus tout à REJECT. */
  get ucpVerdictClass(): string {
    switch (this.ucpDecision) {
      case 'ACCEPT':
      case 'ACCEPT_WITH_FLAGS': return 'PASS';
      case 'REVIEW':
      case 'CONDITIONAL_HOLD':  return 'REVIEW';
      case 'REJECT':            return 'REJECT';
      default:                  return 'REVIEW';
    }
  }

  // ── Verdict final réconcilié (Phase 1) ────────────────────────────────
  /** Décision finale normalisée ; fallback sur UCP puis composite si absente. */
  get finalDecision(): string {
    const d = this.rawFinalDecision?.decision;
    if (d) { return String(d).toUpperCase(); }
    return this.ucpDecision || (this.nlpRoutingDecision || '').toUpperCase();
  }

  /** Libellé lisible du verdict final. */
  get finalDecisionLabel(): string {
    const labels: Record<string, string> = {
      ACCEPT:            '✓ ACCEPTÉ',
      ACCEPT_WITH_FLAGS: '✓ ACCEPTÉ (alertes)',
      REVIEW:            '🟡 REVUE REQUISE',
      CONDITIONAL_HOLD:  '🟠 SOUS RÉSERVE',
      REJECT:            '✕ REJETÉ',
    };
    return labels[this.finalDecision] ?? this.finalDecision;
  }

  /** Classe CSS (PASS / REVIEW / REJECT) du badge verdict final. */
  get finalDecisionClass(): string {
    switch (this.finalDecision) {
      case 'ACCEPT':
      case 'ACCEPT_WITH_FLAGS': return 'PASS';
      case 'REVIEW':
      case 'CONDITIONAL_HOLD':  return 'REVIEW';
      case 'REJECT':            return 'REJECT';
      default:                  return 'REVIEW';
    }
  }

  /** Explique d'où vient le verdict (transparence de la réconciliation). */
  get finalDecisionSourceLabel(): string {
    const labels: Record<string, string> = {
      ucp_hard_block:        'Imposé par blocage UCP (paiement)',
      ucp_contractual_hold:  'Réserve contractuelle UCP (Art.1)',
      ucp_compliance:        'Aligné sur la conformité UCP',
      composite_router:      'Issu du score composite sémantique',
      fallback:              'Revue par défaut',
    };
    return labels[this.rawFinalDecision?.source ?? ''] ?? '';
  }

  // FIX: Only contradictions that have at least message OR a non-empty clause
  get meaningfulContradictions(): UnifiedContradiction[] {
    return this.unifiedContradictions.filter(c =>
      (c.message && c.message !== '—') ||
      (c.clauseA && c.clauseA !== '—') ||
      (c.clauseB && c.clauseB !== '—') ||
      !!c.ucpRef
    );
  }

  // FIX: Filtered meaningful contradictions (respects source filter)
  get filteredMeaningfulContradictions(): UnifiedContradiction[] {
    return this.meaningfulContradictions.filter(c =>
      this.contradictionFilter === 'ALL' || c.source === this.contradictionFilter
    );
  }

  // FIX: Semantic contradictions with real content only
  get meaningfulSemanticContradictions(): UnifiedContradiction[] {
    return this.unifiedContradictions.filter(c =>
      c.source === 'SEMANTIC' &&
      (
        (c.message && c.message !== '—') ||
        (c.clauseA && c.clauseA !== '—') ||
        (c.clauseB && c.clauseB !== '—')
      )
    );
  }

  get semanticContradictionErrors(): number {
    return this.meaningfulSemanticContradictions.filter(c => c.severity === 'error').length;
  }

  get semanticContradictionWarnings(): number {
    return this.meaningfulSemanticContradictions.filter(c => c.severity === 'warning').length;
  }

  // Option B : comptage unifié (NLP + SEMANTIC) pour le bloc Cohérence
  get contradictionErrors(): number {
    return this.meaningfulContradictions.filter(c => /error|critical|high/i.test(c.severity ?? '')).length;
  }

  get contradictionWarnings(): number {
    return this.meaningfulContradictions.filter(c => /warn|medium/i.test(c.severity ?? '')).length;
  }

  get filteredClauses(): RawClause[] {
    const clauses = this.rawNlp?.clauses ?? [];
    return clauses.filter(clause => {
      const matchType = this.clauseFilterType === 'ALL' || (clause.type || '') === this.clauseFilterType;
      const importance = (clause.importance || 'normal').toLowerCase();
      const matchImportance = this.clauseFilterRisk === 'ALL' || importance === this.clauseFilterRisk.toLowerCase();
      return matchType && matchImportance;
    });
  }

  // Keep for backwards compat but redirect to meaningful version
  get filteredContradictions(): UnifiedContradiction[] {
    return this.filteredMeaningfulContradictions;
  }

  get semanticContradictions(): UnifiedContradiction[] {
    return this.meaningfulSemanticContradictions;
  }

  get semanticScoreRows(): SummaryFact[] {
    const record = this.rawSemantic?.coherence_scores ?? {};
    return Object.entries(record).map(([key, value]) => ({
      label: this.humanize(key),
      value: this.renderPercent(value, value <= 1 ? 1 : 100),
      tone:  this.scoreColor(value <= 1 ? value * 100 : value)
    }));
  }

  get enrichedFieldRows(): FeatureRow[] {
    const enriched = this.rawSemantic?.enriched_fields ?? {};
    return Object.entries(enriched).map(([key, value]) => ({
      key:   this.humanize(key),
      value: this.normalizeValue(value)
    }));
  }

  get crossFieldIssues() {
    return this.rawSemantic?.cross_field_coherence?.issues ?? [];
  }

  get nlpRoutingDecision(): string {
    return this.rawSemantic?.routing_decision ?? '';
  }

  get nlpFinalScore(): number {
    return this.rawSemantic?.final_score ?? 0;
  }

  get crossFieldCoherenceScore(): number {
    return this.rawSemantic?.cross_field_coherence?.coherence_score ?? 100;
  }

  get crossFieldVerdict(): string {
    return this.rawSemantic?.cross_field_coherence?.verdict ?? '';
  }

  get crossFieldAxesChecked(): string[] {
    return this.rawSemantic?.cross_field_coherence?.axes_checked ?? [];
  }

  get crossFieldErrors() {
    return this.crossFieldIssues.filter((i: any) => i.severity === 'error');
  }

  get crossFieldWarnings() {
    return this.crossFieldIssues.filter((i: any) => i.severity === 'warning');
  }

  get crossFieldSummary(): { total: number; errors: number; warnings: number; infos: number; blocking: number } {
    const issues = this.crossFieldIssues;
    return {
      total:    issues.length,
      errors:   issues.filter((i: any) => i.severity === 'error').length,
      warnings: issues.filter((i: any) => i.severity === 'warning').length,
      infos:    issues.filter((i: any) => i.severity === 'info').length,
      blocking: issues.filter((i: any) => i.severity === 'error' && (i.confidence ?? 0) >= 0.82).length,
    };
  }

  get crossFieldGroupsByAxis(): CrossFieldGroup[] {
    const axisLabels: Record<string, string> = {
      A: 'Marchandises ↔ Documents',
      B: 'Marchandises ↔ Conditions',
      C: 'Montant ↔ Marchandises',
      D: 'Tags ↔ Conditions (étendu)',
      E: 'Incoterm cross-check',
      F: 'Alignement entités',
      H: 'Quantité ↔ Documents',
    };
    const groups = new Map<string, CrossFieldGroup>();
    for (const issue of this.crossFieldIssues) {
      const code: string = issue.code ?? '';
      // infer axis from code prefix or technique
      let axis = 'Z';
      if (/^GOODS_DOCS|^INCOTERM_CIF_NO_INS|^INCOTERM_MARITIME_NO_BL|^INCOTERM_NON_MARITIME/.test(code))      axis = 'A';
      else if (/^COLD_CHAIN|^BULK_|^PHARMA_/.test(code))                                                       axis = 'B';
      else if (/^AMOUNT_GOODS/.test(code))                                                                      axis = 'C';
      else if (/^PRESENTATION_PERIOD|^CHARGES_TAG|^DRAFT_AT_SIGHT/.test(code))                                 axis = 'D';
      else if (/^INCOTERM/.test(code))                                                                          axis = 'E';
      else if (/^BENEFICIARY_|^APPLICANT_|^AMOUNT_INVOICE|^EXPIRY_|^DRAFT_PAYMENT|^PORT_|^CHARGES_/.test(code)) axis = 'F';
      else if (/^BULK_PACKING|^MULTI_COMMODITY|^HIGH_VALUE/.test(code))                                        axis = 'H';

      if (!groups.has(axis)) {
        groups.set(axis, { axis, label: axisLabels[axis] ?? `Axe ${axis}`, issues: [], errorCount: 0, warningCount: 0 });
      }
      const g = groups.get(axis)!;
      g.issues.push(issue);
      if (issue.severity === 'error')   g.errorCount++;
      if (issue.severity === 'warning') g.warningCount++;
    }
    return [...groups.values()].sort((a, b) => a.axis.localeCompare(b.axis));
  }

  get scoreComponents(): ScoreComponent[] {
    const sem   = this.rawSemantic;
    const cf    = sem?.cross_field_coherence?.coherence_score ?? 100;
    const comp  = sem?.completeness_score ?? (this.analysis?.completenessScore ?? 0);
    const semS  = sem?.semantic_score     ?? (this.analysis?.semanticScore ?? 0);
    const bayes = sem?.bayesian_confidence ?? 1.0;
    const riskFlags: any[] = sem?.risk_flags ?? [];
    const maxRisk = riskFlags.length
      ? Math.max(...riskFlags.map((r: any) => r.severity === 'high' ? 100 : r.severity === 'medium' ? 50 : 20))
      : 0;
    const riskPct = 100 - maxRisk;
    return [
      { label: 'Complétude',       weight: 28, rawScore: comp,         contribution: +(0.28 * comp / 100 * 100).toFixed(1), color: 'blue'   },
      { label: 'Analyse sémantique', weight: 22, rawScore: semS,       contribution: +(0.22 * semS / 100 * 100).toFixed(1), color: 'purple' },
      { label: 'Cohérence champs', weight: 20, rawScore: cf,           contribution: +(0.20 * cf   / 100 * 100).toFixed(1), color: cf >= 80 ? 'green' : cf >= 55 ? 'orange' : 'red' },
      { label: 'Risque absent',    weight: 20, rawScore: riskPct,      contribution: +(0.20 * riskPct / 100 * 100).toFixed(1), color: riskPct >= 80 ? 'green' : 'orange' },
      { label: 'Confiance Bayes',  weight: 10, rawScore: bayes * 100,  contribution: +(0.10 * bayes * 100).toFixed(1), color: 'blue' },
    ];
  }

  techniqueColor(technique: string): string {
    switch ((technique ?? '').toLowerCase()) {
      case 'nli':          return 'blue';
      case 'nli+regex':    return 'purple';
      case 'nli+fuzzy':    return 'purple';
      case 'fuzzy':        return 'soft';
      case 'rule':         return 'green';
      case 'regex':        return 'soft';
      default:             return 'soft';
    }
  }

  confidenceWidth(confidence: number): string {
    return `${Math.round((confidence ?? 0) * 100)}%`;
  }

  get relationRows() { return this.rawNer?.relations ?? []; }

  get externalSources():       string[]             { return this.rawExternal?.sources_consulted ?? []; }
  get externalRiskFactors():   string[]             { return this.rawExternal?.risk_factors ?? []; }
  get externalRecommendations():string[]            { return this.rawExternal?.recommendations ?? []; }
  get externalCountries():     RawExternalCountry[] { return this.rawExternal?.details?.countries ?? []; }
  get externalCurrency():      RawExternalCurrency | null { return this.rawExternal?.details?.currency ?? null; }
  get externalBanks()                               { return this.rawExternal?.details?.banks ?? []; }
  get externalCompanies():     RawExternalCompany[] { return this.rawExternal?.details?.companies ?? []; }

  get riskProbabilities(): Array<{ label: string; value: number; tone: string }> {
    const riskProba = this.rawRiskMl?.risk_proba ?? {};
    return ['LOW', 'MEDIUM', 'HIGH'].map(label => ({
      label,
      value: riskProba[label] ?? 0,
      tone:  label === 'LOW' ? 'green' : label === 'MEDIUM' ? 'orange' : 'red'
    }));
  }

  get featureRows(): FeatureRow[] {
    const features = this.rawRiskMl?.features_used ?? {};
    return Object.entries(features).map(([key, value]) => ({
      key:   this.humanize(key),
      value: this.normalizeValue(value)
    }));
  }

  get shapRows():        RawShapFeature[] { return this.rawRiskMl?.shap_top_features ?? []; }
  get spellCorrections():RawCorrection[]  { return this.rawNlp?.spell?.corrections ?? []; }
  get modulesApplied():  string[]         { return this.rawNlp?.modules_applied ?? []; }

  get hasPerfData(): boolean { return this.perfBars.length > 0 || !!this.rawNlp?.spell; }
  get nlpTotalMs():  number  { return this.rawNlp?.processing_ms ?? 0; }

  get sourceModeLabel(): string {
    const mode = this.rawNer?.source_mode || '';
    if (!mode) { return '—'; }
    return mode === 'rules' ? 'RULES' : 'NER_HIGH_CONF';
  }

  get maxPerfMs(): number {
    const values = this.perfBars.map(i => i.ms);
    return values.length ? Math.max(...values) : 0;
  }

  countClausesByRisk(level: string): number {
    const target = level.toLowerCase();
    return (this.rawNlp?.clauses ?? []).filter(i => (i.risk_level || 'normal').toLowerCase() === target).length;
  }

  mlClauseCount(): number {
    return (this.rawNlp?.clauses ?? []).filter(i => this.clauseMethodLabel(i.method) === 'ML').length;
  }

  timingWidth(ms: number): number {
    if (!this.maxPerfMs) { return 0; }
    return Math.round((ms / this.maxPerfMs) * 100);
  }

  companyFlags(company: RawExternalCompany): string[] {
    const flags: string[] = [];
    if (company.ofac_matched) { flags.push('OFAC'); }
    if (company.eu_matched)   { flags.push('EU'); }
    if (company.un_matched)   { flags.push('UN'); }
    return flags;
  }

  boolText(value: boolean | null | undefined): string {
    if (value == null) { return '—'; }
    return value ? 'Oui' : 'Non';
  }

  trackByAnalysisId   = (_: number, item: AnalysisResponseDTO): number        => item.id;
  trackByPanelId      = (_: number, item: PanelConfig): ValidationPanel        => item.id;
  trackBySwiftTag     = (_: number, item: SwiftFieldRow): string               => item.tag;
  trackByString       = (_: number, item: string): string                      => item;
  trackByContradiction= (_: number, item: UnifiedContradiction): string        => item.id;
  trackByTiming       = (_: number, item: TimingBar): string                   => item.key;
  trackByFeature      = (_: number, item: FeatureRow): string                  => item.key;
  trackByShap         = (_: number, item: RawShapFeature): string              => item.feature || '';

  // ── Private helpers ────────────────────────────────────────

  private syncDerivedData(): void {
    this.analysisPayload    = this.parseRawPayload(this.analysis?.rawPayloadJson);
    this.clauseTypes        = Array.from(
      new Set((this.rawNlp?.clauses ?? []).map(i => i.type || '').filter(Boolean))
    );
    this.swiftFieldRows         = this.buildSwiftFieldRows();
    this.unifiedContradictions  = this.buildUnifiedContradictions();
    this.geoRows                = this.buildGeoRows();
    this.perfBars               = this.buildPerfBars();
    this.shapMaxAbs             = this.buildShapMaxAbs();
    this.validationPanels       = this.buildPanels();

    if (!this.clauseTypes.includes(this.clauseFilterType)) {
      this.clauseFilterType = 'ALL';
    }
  }

  private parseRawPayload(rawPayloadJson: string | null | undefined): RawAnalysisPayload | null {
    if (!rawPayloadJson) { return null; }
    try   { return JSON.parse(rawPayloadJson) as RawAnalysisPayload; }
    catch { return null; }
  }

  private buildSwiftFieldRows(): SwiftFieldRow[] {
    const validationFields = this.rawValidation?.fields ?? {};
    const rawFields        = (this.rawSwift?.entities?.raw_fields ?? {}) as Record<string, string>;
    const seen             = new Set<string>();
    const rows: SwiftFieldRow[] = [];
    const blueprintOrder   = new Map<string, number>();

    SWIFT_FIELD_BLUEPRINT.forEach((item, index) => {
      blueprintOrder.set(item.tag.replace('-BY', ''), index);
      const row = this.buildSwiftFieldRow(item.tag, item.key, item.label, validationFields, rawFields);
      if (row) {
        rows.push(row);
        seen.add(this.cleanTag(item.tag));
      }
    });

    // Add remaining fields from validationFields, excluding Additional Conditions
    Object.keys(validationFields).forEach(tag => {
      const clean = this.cleanTag(tag);
      if (!seen.has(clean) && !EXCLUDED_SWIFT_TAGS.has(clean)) {
        const row = this.buildSwiftFieldRow(clean, '', undefined, validationFields, rawFields);
        if (row) {
          rows.push(row);
          seen.add(clean);
        }
      }
    });

    // Add remaining fields from rawFields, excluding Additional Conditions
    Object.keys(rawFields).forEach(tag => {
      const clean = this.cleanTag(tag);
      if (!seen.has(clean) && !EXCLUDED_SWIFT_TAGS.has(clean)) {
        const row = this.buildSwiftFieldRow(clean, '', undefined, validationFields, rawFields);
        if (row) {
          rows.push(row);
          seen.add(clean);
        }
      }
    });

    return rows.sort((l, r) => {
      const lOrder = blueprintOrder.get(this.cleanTag(l.tag));
      const rOrder = blueprintOrder.get(this.cleanTag(r.tag));
      if (lOrder != null && rOrder != null) { return lOrder - rOrder; }
      if (lOrder != null)  { return -1; }
      if (rOrder != null)  { return  1; }
      return l.tag.localeCompare(r.tag, 'fr');
    });
  }

  private buildSwiftFieldRow(
    tag: string,
    key: string,
    fallbackLabel: string | undefined,
    validationFields: Record<string, RawValidationField>,
    rawFields: Record<string, string>
  ): SwiftFieldRow | null {
    const entities  = this.rawSwift?.entities ?? ({} as RawSwiftEntities);
    const cleanTag  = this.cleanTag(tag);

    // FIX: Skip Additional Conditions tag
    if (EXCLUDED_SWIFT_TAGS.has(cleanTag)) { return null; }

    const validationField =
      validationFields[cleanTag] ||
      validationFields[`:${cleanTag}:`] ||
      validationFields[tag] ||
      undefined;

    const rawValue       = rawFields[cleanTag] || rawFields[`:${cleanTag}:`] || rawFields[tag] || '';
    const structuredValue= key ? entities[key] : undefined;
    const enrichedFields = this.rawSemantic?.enriched_fields ?? {};
    const inferredValue  = key ? enrichedFields[`${key}_inferred`] ?? enrichedFields[key] : undefined;

    let value: unknown = structuredValue ?? rawValue ?? validationField?.value ?? null;
    const sourceBadges: string[] = [];

    if (!this.hasValue(rawValue) && this.hasValue(inferredValue)) {
      value = inferredValue;
      sourceBadges.push('🔮 INF');
    }

    const sourceKey   = key ? `_${key}_source` : '';
    const sourceValue = sourceKey ? entities[sourceKey] : undefined;
    if ((!this.hasValue(rawValue) && this.hasValue(value)) || typeof sourceValue === 'string') {
      if (!sourceBadges.includes('🤖 NLP')) { sourceBadges.push('🤖 NLP'); }
    }

    const errors    = validationField?.errors   ?? [];
    const warnings  = validationField?.warnings ?? [];
    const mandatory = !!validationField?.mandatory ||
      (this.rawValidation?.missing_mandatory ?? []).includes(cleanTag);

    const displayValue = key === 'amount' && this.analysis?.amount != null
      ? this.formatAmount(this.analysis.amount, this.analysis.currency)
      : this.normalizeValue(value);

    const confidence = this.findFieldConfidence(key, displayValue);
    if (
      confidence != null &&
      !sourceBadges.includes('🤖 NLP') &&
      !sourceBadges.includes('🔮 INF') &&
      !this.hasValue(rawValue)
    ) {
      sourceBadges.push('🤖 NLP');
    }

    const status: SwiftFieldRow['status'] =
      errors.length > 0 || (mandatory && displayValue === '—')
        ? 'error'
        : warnings.length > 0
          ? 'warn'
          : displayValue !== '—'
            ? 'ok'
            : 'neutral';

    if (displayValue === '—' && !errors.length && !warnings.length && !mandatory) {
      return null;
    }

    return {
      tag:          `:${cleanTag}:`,
      name:         validationField?.name || fallbackLabel || this.humanize(key || cleanTag),
      value:        displayValue,
      status,
      mandatory,
      sourceBadges,
      confidence,
      errors,
      warnings
    };
  }

  private buildUnifiedContradictions(): UnifiedContradiction[] {
    // Source canonique : nlp.contradictions (fusion NLP + sémantique taguée par
    // le pipeline). Chaque item porte sa propre source.
    const topRows = (this.rawNlp?.contradictions ?? []).map((item, i) =>
      this.mapContradiction(item, item.source === 'SEMANTIC' ? 'SEMANTIC' : 'NLP', i)
    );
    // Rétro-compat : anciens payloads où les contradictions tag↔clause ne vivent
    // QUE dans semantic.contradictions (pas encore fusionnées dans le top-level).
    const semRows = (this.rawSemantic?.contradictions ?? []).map((item, i) =>
      this.mapContradiction(item, 'SEMANTIC', i + 100000)
    );
    // Déduplication par signature (évite de compter 2× une sémantique présente
    // à la fois dans le top-level fusionné et dans semantic.contradictions).
    const seen = new Set<string>();
    const out: UnifiedContradiction[] = [];
    for (const c of [...topRows, ...semRows]) {
      const key = `${c.source}|${c.clauseA}|${c.clauseB}|${c.message}`;
      if (seen.has(key)) { continue; }
      seen.add(key);
      out.push(c);
    }
    return out;
  }

  private buildGeoRows(): GeoRow[] {
    const geo = this.rawSemantic?.geo_coherence;
    if (!geo) { return []; }

    if (geo === true) {
      return [{ key: 'geo_ok', status: 'ok', country: '—', location: 'Cohérence globale', risk: 'LOW', details: 'Cohérence géographique OK.' }];
    }

    if (Array.isArray(geo)) {
      return geo.map((item, i) => this.mapGeoEntry(String(i), item));
    }

    if (typeof geo === 'object') {
      const g = geo as Record<string, unknown>;
      const rows: GeoRow[] = [];
      const beneficiary = this.analysis?.beneficiary ?? '';

      const isoToLabel: Record<string, string> = {
        FR: 'France', TN: 'Tunisie', DE: 'Allemagne', GB: 'Royaume-Uni',
        US: 'États-Unis', CN: 'Chine', TR: 'Turquie', MA: 'Maroc',
        DZ: 'Algérie', EG: 'Égypte', AE: 'Émirats Arabes Unis', SA: 'Arabie Saoudite',
        IT: 'Italie', ES: 'Espagne', NL: 'Pays-Bas', BE: 'Belgique',
        CH: 'Suisse', IN: 'Inde', JP: 'Japon', KR: 'Corée du Sud',
        SG: 'Singapour', HK: 'Hong Kong', SN: 'Sénégal', NG: 'Nigéria',
        LY: 'Libye', IR: 'Iran', PK: 'Pakistan',
      };
      const resolveCode = (code: string) => isoToLabel[code?.toUpperCase()] ?? code ?? '—';

      // ── A : expiry place ─────────────────────────────────────
      if ('expiry_place' in g) {
        const raw  = String(g['expiry_place'] ?? '').toUpperCase();
        const iso  = String(g['expiry_country_code'] ?? raw).toUpperCase();
        const label = resolveCode(iso);
        rows.push({
          key: 'expiry_place',
          status: 'info',
          country: label,
          location: 'Lieu d\'expiration (:31D:)',
          risk: 'LOW',
          details: iso ? `Code ISO: ${iso}` : raw || 'Non renseigné',
        });
      }

      // ── B : expiry vs bénéficiaire ───────────────────────────
      if ('expiry_beneficiary_match' in g) {
        const match    = g['expiry_beneficiary_match'] === true;
        const expIso   = String(g['expiry_country_code']  ?? g['expiry_place'] ?? '').toUpperCase();
        const beneIso  = String(g['beneficiary_country_code'] ?? '').toUpperCase();
        const expLabel  = resolveCode(expIso);
        const beneLabel = beneIso ? resolveCode(beneIso) : beneficiary;
        rows.push({
          key: 'expiry_beneficiary_match',
          status: match ? 'ok' : 'warn',
          country: expLabel || '—',
          location: 'Expiration vs pays bénéficiaire',
          risk: match ? 'LOW' : 'MEDIUM',
          details: match
            ? `Cohérent — lieu d'expiration (${expLabel}) correspond au pays bénéficiaire (${beneLabel})`
            : `⚠ Lieu d'expiration (${expLabel}${expIso ? ' · ' + expIso : ''}) ≠ pays bénéficiaire (${beneLabel}${beneIso ? ' · ' + beneIso : ''})`,
        });
      }

      // ── C : port chargement ──────────────────────────────────
      if ('port_loading' in g && g['port_loading']) {
        const val = String(g['port_loading']);
        const iso = String(g['port_loading_country_code'] ?? '').toUpperCase();
        rows.push({
          key: 'port_loading',
          status: 'info',
          country: iso ? resolveCode(iso) : val,
          location: 'Port de chargement (:44A:)',
          risk: 'LOW',
          details: iso ? `${val} · ISO: ${iso}` : val,
        });
      }

      // ── D : port déchargement vs bénéficiaire ────────────────
      if ('port_discharge' in g && g['port_discharge']) {
        const val   = String(g['port_discharge']);
        const dIso  = String(g['port_discharge_country_code'] ?? g['port_discharge'] ?? '').toUpperCase();
        const bIso  = String(g['beneficiary_country_code'] ?? '').toUpperCase();
        const mismatch = 'port_discharge_match' in g && g['port_discharge_match'] === false;
        rows.push({
          key: 'port_discharge',
          status: mismatch ? 'warn' : 'info',
          country: dIso ? resolveCode(dIso) : val,
          location: 'Port de déchargement (:44B:)',
          risk: mismatch ? 'MEDIUM' : 'LOW',
          details: mismatch
            ? `⚠ Port (${resolveCode(dIso) || val}) ne correspond pas au pays bénéficiaire (${resolveCode(bIso) || bIso})`
            : dIso ? `${val} · ISO: ${dIso}` : val,
        });
      }

      // Champs inconnus restants → fallback générique
      const known = new Set([
        'expiry_place','expiry_beneficiary_match','port_loading','port_discharge',
        'expiry_country_code','beneficiary_country_code',
        'port_loading_country_code','port_discharge_country_code','port_discharge_match',
      ]);
      for (const [key, value] of Object.entries(g)) {
        if (known.has(key)) { continue; }
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          rows.push(this.mapGeoEntry(key, value as RawGeoEntry));
        } else {
          rows.push({ key, status: this.inferGeoStatus(value), country: '—', location: this.humanize(key), risk: this.inferGeoRisk(value), details: this.normalizeValue(value) });
        }
      }

      return rows;
    }

    return [];
  }

  private buildPerfBars(): TimingBar[] {
    const timings = this.rawNlp?.module_timings ?? {};
    const bars: TimingBar[] = Object.entries(timings).map(([key, value]) => ({
      key, label: this.humanize(key), ms: Math.round(value), tone: 'blue'
    }));

    const extra = [
      { key: 'ucp',      label: 'UCP',      ms: this.rawUcp?.processing_ms,      tone: 'orange' },
      { key: 'external', label: 'External', ms: this.rawExternal?.processing_ms,  tone: 'orange' },
      { key: 'ds',       label: 'DS',       ms: this.rawDs?.processing_ms,        tone: 'purple' },
      { key: 'risk_ml',  label: 'Risk ML',  ms: this.rawRiskMl?.processing_ms,    tone: 'purple' }
    ];

    extra.forEach(item => {
      if (item.ms != null && item.ms > 0 && !bars.find(b => b.key === item.key)) {
        bars.push({ key: item.key, label: item.label, ms: Math.round(item.ms), tone: item.tone });
      }
    });

    return bars.sort((l, r) => r.ms - l.ms);
  }

  private buildShapMaxAbs(): number {
    const values = (this.rawRiskMl?.shap_top_features ?? []).map(i => Math.abs(i.shap_value ?? 0));
    return values.length ? Math.max(...values) : 0;
  }

  private buildPanels(): PanelConfig[] {
    const clausesCount      = this.rawNlp?.clauses?.length ?? this.analysis?.nClauses ?? 0;
    const validationCount   = (this.rawValidation?.missing_mandatory?.length ?? 0) +
                               (this.rawValidation?.cross_field_errors?.length ?? 0);
    const ucpCount          = this.ucpErrorCount + this.ucpWarningCount;
    const coherenceCount    = (this.rawSemantic?.risk_flags?.length ?? 0) +
                               (this.rawSemantic?.warnings?.length ?? 0) +
                               (this.rawSemantic?.cross_field_coherence?.issues?.length ?? 0);
    const externalCount     = (this.rawExternal?.risk_factors?.length ?? 0) +
                               (this.externalCompanies.length || 0);
    const riskMlCount       = this.rawRiskMl?.shap_top_features?.length ?? 0;

    return [
      { id: 'overview',    icon: '◈',  label: 'Synthèse' },
      { id: 'swift-fields',icon: '📋', label: 'Champs du crédit', count: this.swiftFieldRows.length },
      { id: 'clauses',     icon: '📌', label: 'Clauses & conditions', count: clausesCount },
      { id: 'validation',  icon: '✅', label: 'Validation',     count: validationCount },
      { id: 'ucp',         icon: '⚖️', label: 'Conformité UCP 600', count: ucpCount },
      { id: 'coherence',   icon: '🔬', label: 'Cohérence',      count: coherenceCount },
      { id: 'external',    icon: '🌐', label: 'Sanctions & pays', count: externalCount },
      { id: 'risk-ml',     icon: '⚡', label: 'Score de risque', count: riskMlCount }
    ];
  }

  private findFieldConfidence(fieldKey: string, displayValue: string): number | null {
    if (!fieldKey || !this.rawNer?.entities?.length) { return null; }

    const labels          = NER_LABELS_BY_FIELD[fieldKey] ?? [];
    const normalizedDisplay = displayValue.toLowerCase();
    const matches = this.rawNer.entities.filter(entity => {
      const entityLabel = (entity.label || '').toUpperCase();
      if (!labels.includes(entityLabel)) { return false; }
      const text = `${entity.value || ''} ${entity.normalized || ''}`.toLowerCase();
      return !normalizedDisplay || normalizedDisplay === '—' ||
             text.includes(normalizedDisplay) || normalizedDisplay.includes(text.trim());
    });

    const selected = matches.length ? matches
      : this.rawNer.entities.filter(e => labels.includes((e.label || '').toUpperCase()));

    if (!selected.length) { return null; }
    return Math.max(...selected.map(e => e.confidence ?? 0));
  }

  private mapContradiction(
    item: RawContradiction,
    source: 'NLP' | 'SEMANTIC',
    index: number
  ): UnifiedContradiction {
    const isSemanticEngine = !!(item.tag_field || item.tag_value || item.clause_text);

    let message: string;
    let clauseA: string;
    let clauseB: string;

    if (isSemanticEngine) {
      // ContradictionEngine format: tag_field / tag_value / clause_text / recommendation
      const tag   = item.tag_field  ? `:${item.tag_field?.toUpperCase()}: = ${item.tag_value ?? '?'}` : '';
      const rec   = item.recommendation ?? item.clause_intent ?? '';
      message  = rec || tag || '—';
      clauseA  = tag  || '—';
      clauseB  = this.normalizeValue(item.clause_text ?? '—');
    } else {
      // NLP / cross-field format: tolère plusieurs noms de champs back-end
      message = this.normalizeValue(
        item.message ??
        (item['description'] as string | undefined) ??
        (item['detail']      as string | undefined) ??
        (item['reason']      as string | undefined) ??
        (item['explanation'] as string | undefined) ??
        (item['suggestion']  as string | undefined) ??
        (item['code']        as string | undefined)
      );
      clauseA = this.normalizeValue(
        item.clause_a ??
        (item['field']    as string | undefined) ??
        (item['lc_value'] as string | undefined) ??
        (Array.isArray(item['fields']) ? (item['fields'] as string[]).join(' ↔ ') : undefined) ??
        '—'
      );
      clauseB = this.normalizeValue(
        item.clause_b ??
        (item['doc_value'] as string | undefined) ??
        (item['detail']    as string | undefined) ??
        '—'
      );
    }

    const severity = this.normalizeValue(item.severity || (item['level'] as string | undefined) || 'warning');
    const ucpRef   = item.ucp_basis ?? (item['ucp_ref'] as string | undefined) ?? '';
    const conf     = item.confidence ?? (item['score'] as number | undefined);

    // Si le back-end n'a fourni aucun libellé exploitable mais qu'il existe bien
    // une donnée (clause/tag/ucp), on synthétise un message pour ne pas perdre la
    // contradiction au filtrage « meaningful » (cause classique du compteur à 0).
    if (message === '—') {
      if (clauseA !== '—' && clauseB !== '—') { message = `Incohérence entre ${clauseA} et ${clauseB}`; }
      else if (clauseA !== '—')               { message = `Incohérence détectée sur ${clauseA}`; }
      else if (ucpRef)                         { message = `Contradiction réglementaire (${ucpRef})`; }
    }

    return {
      id: `${source}_${index}_${message}`,
      source,
      severity,
      message,
      clauseA,
      clauseB,
      ucpRef,
      confidence: conf,
    };
  }

  private mapGeoEntry(key: string, entry: RawGeoEntry): GeoRow {
    return {
      key,
      status:   String(entry.status || entry['state'] || this.inferGeoStatus(entry.details || entry.risk || key)),
      country:  String(entry.country || entry['iso2'] || '—'),
      location: String(entry.port || entry['location'] || this.humanize(key)),
      risk:     String(entry.risk || this.inferGeoRisk(entry.status || entry.details)),
      details:  this.normalizeValue(entry.details || entry['reason'] || entry)
    };
  }

  private inferGeoStatus(value: unknown): string {
    const text = this.normalizeValue(value).toLowerCase();
    if (text.includes('error') || text.includes('mismatch')) { return 'error'; }
    if (text.includes('warn'))                               { return 'warning'; }
    if (text === 'true' || text.includes('ok'))              { return 'ok'; }
    return 'info';
  }

  private inferGeoRisk(value: unknown): string {
    const text = this.normalizeValue(value).toLowerCase();
    if (text.includes('high')   || text.includes('error')) { return 'HIGH'; }
    if (text.includes('medium') || text.includes('warn'))  { return 'MEDIUM'; }
    if (text.includes('low')    || text.includes('ok') || text === 'true') { return 'LOW'; }
    return 'INFO';
  }

  private cleanTag(tag: string): string {
    return tag.replaceAll(':', '').replace('-BY', '').trim().toUpperCase();
  }

  private hasValue(value: unknown): boolean {
    if (value == null)             { return false; }
    if (typeof value === 'string') { return value.trim().length > 0; }
    return true;
  }

  private humanize(value: string): string {
    return value
      .replaceAll('_', ' ')
      .replaceAll('-', ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .replace(/\b\w/g, c => c.toUpperCase());
  }

  private basePanels(): PanelConfig[] {
    return [
      { id: 'overview',    icon: '◈',  label: 'Synthèse' },
      { id: 'swift-fields',icon: '📋', label: 'Champs du crédit' },
      { id: 'clauses',     icon: '📌', label: 'Clauses & conditions' },
      { id: 'validation',  icon: '✅', label: 'Validation' },
      { id: 'ucp',         icon: '⚖️', label: 'Conformité UCP 600' },
      { id: 'coherence',   icon: '🔬', label: 'Cohérence' },
      { id: 'external',    icon: '🌐', label: 'Sanctions & pays' },
      { id: 'risk-ml',     icon: '⚡', label: 'Score de risque' }
    ];
  }
}
