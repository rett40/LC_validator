import {
  Component, OnInit, ChangeDetectorRef, signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, ActivatedRoute } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';
import { CommercialService } from '../../core/services/commercial.service';
import { ReportService, ReportMeta } from '../../core/services/report.service';
import {
  CommercialValidationResponseDTO,
  CommercialDocumentResult,
  CommercialRuleResult,
  CrossExplanation,
  CrossScoreBreakdown,
} from '../../core/models/commercial.model';
import { LcCrossValidationComponent } from './lc-cross-validation.component';

type CrossTab = 'rules' | 'semantic' | 'nli' | 'tables' | 'xai';
type DocTab    = 'fields' | 'tables' | 'rules';

const DOC_LABELS: Record<string, string> = {
  bl:           'Bill of Lading',
  invoice:      'Invoice',
  packing_list: 'Packing List',
};

const INPUT_LABELS: Record<string, string> = {
  pdf_native:  'PDF natif',
  pdf_scanned: 'PDF scanné',
  image:       'Image',
};

const RISK_COLORS: Record<string, { bg: string; fg: string }> = {
  low:      { bg: '#DCFCE7', fg: '#166534' },
  medium:   { bg: '#FEF3C7', fg: '#92400E' },
  high:     { bg: '#FEE2E2', fg: '#991B1B' },
  critical: { bg: '#FEE2E2', fg: '#DC2626' },
  unknown:  { bg: '#F1F5F9', fg: '#64748B' },
};

@Component({
  selector: 'app-commercial-docs',
  standalone: true,
  imports: [CommonModule, FormsModule, LcCrossValidationComponent],
  templateUrl: './commercial-docs.component.html',
  styleUrls: ['./commercial-docs.component.scss'],
})
export class CommercialDocsComponent implements OnInit {
  readonly maxFiles = 3;
  readonly acceptedTypes = '.pdf,.png,.jpg,.jpeg,.tiff,.tif,.bmp';

  isDragging  = false;
  selectedFiles: File[] = [];
  uploading   = false;
  uploadError = '';
  uiState: 'idle' | 'uploading' | 'success' | 'error' = 'idle';

  lcReference = '';
  swiftAnalysisId: number | undefined;
  result: CommercialValidationResponseDTO | null = null;

  activeCrossTab: CrossTab = 'rules';
  activeDocIdx   = 0;
  activeDocTab: DocTab = 'fields';

  statusFilter: string = 'ALL';

  history = signal<CommercialValidationResponseDTO[]>([]);
  historyLoading = signal(false);
  showHistory    = false;

  showLcCross = false;

  // ── Rapport complet PDF ──────────────────────────────────────────
  fullReport: ReportMeta | null = null;
  reportBusy = false;
  reportError = '';

  constructor(
    private readonly commercialService: CommercialService,
    private readonly router: Router,
    private readonly route: ActivatedRoute,
    private readonly cdr: ChangeDetectorRef,
    private readonly reportService: ReportService,
  ) {}

  /** Le rapport complet n'est possible que si la validation est liée à un MT700. */
  get canGenerateFull(): boolean {
    return this.swiftAnalysisId != null && this.result != null;
  }

  loadFullReport(): void {
    if (this.swiftAnalysisId == null) { return; }
    this.reportService.listReports(this.swiftAnalysisId).subscribe({
      next: (reports) => {
        this.fullReport = (reports ?? []).find(r => r.reportType === 'FULL_REPORT') ?? null;
        this.cdr.markForCheck();
      },
      error: () => {},
    });
  }

  generateFullReport(): void {
    if (this.swiftAnalysisId == null) { return; }
    this.reportBusy = true;
    this.reportError = '';
    // L'id de la validation commerciale courante permet d'inclure la
    // cross-validation inter-documents et la cohérence des documents avec le MT700.
    this.reportService.generateFull(this.swiftAnalysisId, this.result?.id).subscribe({
      next: (r) => {
        this.fullReport = r;
        this.reportBusy = false;
        this.reportService.openPdf(r);
        this.cdr.markForCheck();
      },
      error: () => {
        this.reportBusy = false;
        this.reportError = 'Échec de génération du rapport complet. Réessayez.';
        this.cdr.markForCheck();
      },
    });
  }

  openFullReport(): void {
    if (this.fullReport) { this.reportService.openPdf(this.fullReport); }
  }

  ngOnInit(): void {
    const id = this.route.snapshot.queryParamMap.get('swiftAnalysisId');
    if (id) { this.swiftAnalysisId = Number(id); this.loadFullReport(); }
    this.loadHistory();
  }

  loadHistory(): void {
    this.historyLoading.set(true);
    // Historique aligné sur le MT700 courant : quand la vue est ouverte depuis une
    // analyse SWIFT, ne charger que les validations rattachées à ce MT700. Sinon,
    // liste globale. Cap relevé à 50 (au lieu des 10 précédents).
    this.commercialService.list(0, 50, undefined, undefined, this.swiftAnalysisId).subscribe({
      next: res => {
        this.history.set(res?.content ?? []);
        this.historyLoading.set(false);
      },
      error: () => {
        this.history.set([]);
        this.historyLoading.set(false);
      },
    });
  }

  // ── Drag & drop ──────────────────────────────────────────────────────

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
    const files = Array.from(event.dataTransfer?.files ?? []);
    this.addFiles(files);
  }

  onFileChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    const files = Array.from(input?.files ?? []);
    this.addFiles(files);
    input.value = '';
  }

  private addFiles(files: File[]): void {
    const remaining = this.maxFiles - this.selectedFiles.length;
    const toAdd = files.slice(0, remaining);
    this.selectedFiles = [...this.selectedFiles, ...toAdd];
    if (files.length > remaining) {
      this.uploadError = `Maximum ${this.maxFiles} documents. ${files.length - remaining} ignoré(s).`;
    } else {
      this.uploadError = '';
    }
  }

  removeFile(idx: number): void {
    this.selectedFiles = this.selectedFiles.filter((_, i) => i !== idx);
  }

  // ── Analysis ─────────────────────────────────────────────────────────

  analyze(): void {
    if (this.selectedFiles.length === 0 || this.uploading) return;
    this.uploading   = true;
    this.uiState     = 'uploading';
    this.uploadError = '';
    this.result      = null;

    this.commercialService
      .validate(this.selectedFiles, this.lcReference || undefined, this.swiftAnalysisId)
      .subscribe({
        next: (res: CommercialValidationResponseDTO) => {
          this.result = res;
          this.uploading = false;
          this.uiState   = 'success';
          this.activeCrossTab = 'rules';
          this.activeDocIdx   = 0;
          this.activeDocTab   = 'fields';
          this.selectedFiles  = [];
          this.updateTableEntries();
          this.loadFullReport();
          this.cdr.detectChanges();
          this.loadHistory();
        },
        error: (err: HttpErrorResponse) => {
          if (err.status === 401 || err.status === 403) {
            this.router.navigateByUrl('/login');
          } else if (err.status === 0) {
            this.uploadError = 'Backend indisponible. Vérifiez les services.';
          } else {
            this.uploadError = err.error?.message || 'Erreur lors de la validation.';
          }
          this.uploading = false;
          this.uiState   = 'error';
        },
      });
  }

  loadFromHistory(item: CommercialValidationResponseDTO): void {
    this.showHistory = false;
    this.result = item;
    this.uiState = 'success';
    this.activeCrossTab = 'rules';
    this.activeDocIdx = 0;
    this.activeDocTab = 'fields';
    this.updateTableEntries();
    if (!item.fullPayloadJson) {
      this.commercialService.getById(item.id).subscribe({
        next: res => {
          this.result = res;
          this.updateTableEntries();
          this.cdr.detectChanges();
        },
        error: () => {},
      });
    }
  }

  goBack(): void {
    this.router.navigateByUrl('/app');
  }

  reset(): void {
    this.result      = null;
    this.uiState     = 'idle';
    this.uploadError = '';
    this.selectedFiles = [];
    this.lcReference = '';
  }

  // ── Display helpers ──────────────────────────────────────────────────

  docLabel(docType: string): string {
    return DOC_LABELS[docType] || docType?.toUpperCase() || '—';
  }

  inputLabel(inputType: string): string {
    return INPUT_LABELS[inputType] || inputType || '—';
  }

  riskStyle(level: string | null): { bg: string; fg: string } {
    const key = (level || 'unknown').toLowerCase();
    return RISK_COLORS[key] || RISK_COLORS['unknown'];
  }

  riskIcon(level: string | null): string {
    const k = (level || '').toLowerCase();
    if (k === 'low')      return '🟢';
    if (k === 'medium')   return '🟡';
    if (k === 'high')     return '🔴';
    if (k === 'critical') return '🚨';
    return '⚪';
  }

  confColor(conf: number): string {
    if (conf >= 0.80) return '#16A34A';
    if (conf >= 0.60) return '#F59E0B';
    return '#DC2626';
  }

  scoreColor(score: number | null | undefined): string {
    const v = score ?? 0;
    if (v >= 0.75) return '#16A34A';
    if (v >= 0.55) return '#F59E0B';
    return '#DC2626';
  }

  scorePercent(v: number | null | undefined): string {
    if (v == null) return '—';
    return `${Math.round(v * 100)}%`;
  }

  scoreWidth(v: number | null | undefined): number {
    return Math.round((v ?? 0) * 100);
  }

  statusIcon(status: string): string {
    if (status === 'pass')      return '✅';
    if (status === 'fail')      return '❌';
    if (status === 'warning')   return '⚠️';
    if (status === 'corrected') return '🔧';
    return '⏭️';
  }

  statusBorderColor(status: string): string {
    if (status === 'pass')    return '#16A34A';
    if (status === 'fail')    return '#DC2626';
    if (status === 'warning') return '#F59E0B';
    return '#E2E8F0';
  }

  statusBg(status: string): string {
    if (status === 'pass')    return '#DCFCE733';
    if (status === 'fail')    return '#FEE2E233';
    if (status === 'warning') return '#FEF3C733';
    return '#F1F5F933';
  }

  sevStyle(sev: string): string {
    if (sev === 'critical') return 'background:#FEE2E2;color:#DC2626';
    if (sev === 'high')     return 'background:#FEE2E2;color:#991B1B';
    if (sev === 'medium')   return 'background:#FEF3C7;color:#92400E';
    if (sev === 'low')      return 'background:#DCFCE7;color:#166534';
    return 'background:#F1F5F9;color:#64748B';
  }

  nliIcon(label: string): string {
    if (label === 'ENTAILMENT')    return '✅';
    if (label === 'CONTRADICTION') return '❌';
    return '➖';
  }

  nliColor(label: string): string {
    if (label === 'ENTAILMENT')    return '#DCFCE7';
    if (label === 'CONTRADICTION') return '#FEE2E2';
    return '#F1F5F9';
  }

  impactIcon(impact: string): string {
    if (impact === 'positive') return '✅';
    if (impact === 'negative') return '❌';
    if (impact === 'warning')  return '⚠️';
    return 'ℹ️';
  }

  impactBorder(impact: string): string {
    if (impact === 'positive') return '#16A34A';
    if (impact === 'negative') return '#DC2626';
    if (impact === 'warning')  return '#F59E0B';
    return '#E2E8F0';
  }

  tierStyle(tier: string): string {
    const t = (tier || '').toLowerCase();
    if (t.startsWith('spatial'))         return 'background:#DCFCE7;color:#166534';
    if (t === 'layout_structure')        return 'background:#DCFCE7;color:#166534';
    if (t === 'regex' || t === 'kv_regex') return 'background:#FEF3C7;color:#92400E';
    if (t === 'semantic' || t.startsWith('semantic')) return 'background:#E8F0FE;color:#1A3A6E';
    if (t.startsWith('zone_'))           return 'background:#E8F0FE;color:#1A3A6E';
    if (t === 'layoutlm')                return 'background:#E8F0FE;color:#2C5FA3';
    if (t === 'computed')                return 'background:#FEF3C7;color:#B45309';
    if (t === 'fallback_critical')       return 'background:#FEE2E2;color:#991B1B';
    return 'background:#F1F5F9;color:#64748B';
  }

  get filteredRules(): CommercialRuleResult[] {
    const rules = this.result?.crossReport?.ruleResults ?? [];
    if (this.statusFilter === 'ALL') return rules;
    return rules.filter(r => r.status === this.statusFilter);
  }

  get filteredDocRules(): CommercialRuleResult[] {
    const doc = this.activeDoc;
    if (!doc) return [];
    return doc.ruleResults ?? [];
  }

  get activeDoc(): CommercialDocumentResult | null {
    return this.result?.documents?.[this.activeDocIdx] ?? null;
  }

  get fieldEntries(): [string, string][] {
    const f = this.activeDoc?.fields ?? {};
    return Object.entries(f).sort(([a], [b]) => a.localeCompare(b));
  }

  tableEntries: [string, unknown[]][] = [];

  updateTableEntries(): void {
    const doc = this.activeDoc;
    if (!doc?.tables) { this.tableEntries = []; return; }
    this.tableEntries = Object.entries(doc.tables)
      .filter(([, v]) => Array.isArray(v) && (v as unknown[]).length > 0) as [string, unknown[]][];
  }

  tableColumns(rows: unknown[]): string[] {
    const all = new Set<string>();
    (rows as Record<string, unknown>[]).forEach(r => Object.keys(r).forEach(k => all.add(k)));
    return Array.from(all);
  }

  tableRowValue(row: unknown, col: string): string {
    const r = row as Record<string, unknown>;
    const v = r[col];
    if (v == null) return '';
    return String(v);
  }

  semanticEntries(): [string, number][] {
    const sc = this.result?.crossReport?.semanticScores ?? {};
    return Object.entries(sc)
      .filter(([, v]) => typeof v === 'number')
      .sort(([, a], [, b]) => (b as number) - (a as number)) as [string, number][];
  }

  semanticLabel(key: string): string {
    if (key.includes(':')) {
      const [pair, sub] = key.split(':', 2);
      return `${pair} [${sub}]`;
    }
    return key;
  }

  breakdownEntries(): [string, number][] {
    const bd = this.result?.crossReport?.scoreBreakdown ?? {} as CrossScoreBreakdown;
    const order = ['semantic', 'nli', 'rules', 'table', 'party'];
    return order
      .filter(k => (bd as Record<string, number>)[k] != null)
      .map(k => [k, (bd as Record<string, number>)[k]] as [string, number]);
  }

  breakdownLabel(k: string): string {
    const labels: Record<string, string> = {
      semantic: 'Semantic Embedding',
      nli: 'NLI Inference',
      rules: 'Business Rules',
      table: 'Table Alignment',
      party: 'Party Matching',
    };
    return labels[k] || k;
  }

  breakdownColor(k: string): string {
    const colors: Record<string, string> = {
      semantic: '#2C5FA3',
      nli: '#EC4899',
      rules: '#F59E0B',
      table: '#16A34A',
      party: '#1A3A6E',
    };
    return colors[k] || '#6B7280';
  }

  breakdownWeight(k: string): string {
    const weights: Record<string, number> = {
      semantic: 30, nli: 35, rules: 25, table: 10, party: 10,
    };
    return weights[k] != null ? `${weights[k]}%` : '';
  }

  formatDate(d: string | null): string {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('fr-FR', {
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  }

  historyRiskClass(level: string | null): string {
    const k = (level || '').toLowerCase();
    if (k === 'low')      return 'risk-low';
    if (k === 'medium')   return 'risk-medium';
    if (k === 'high')     return 'risk-high';
    if (k === 'critical') return 'risk-critical';
    return 'risk-unknown';
  }

  fileIcon(file: File): string {
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    if (ext === 'pdf') return '📄';
    return '🖼️';
  }

  fileSizeLabel(file: File): string {
    if (file.size < 1024) return `${file.size} B`;
    if (file.size < 1024 * 1024) return `${Math.round(file.size / 1024)} Ko`;
    return `${(file.size / (1024 * 1024)).toFixed(1)} Mo`;
  }

  countRulesByStatus(rules: CommercialRuleResult[], status: string): number {
    return (rules ?? []).filter(r => r.status === status).length;
  }

  countNliByLabel(results: { nli_label: string }[], label: string): number {
    return (results ?? []).filter(r => r.nli_label === label).length;
  }

  filterExplanationsByImpact(explanations: CrossExplanation[], impact: string): CrossExplanation[] {
    return (explanations ?? []).filter(e => e.impact === impact);
  }

  trackByIdx = (i: number): number => i;
  trackByKey = (_: number, item: [string, unknown]): string => item[0];
}
