import {
  Component, Input, Output, EventEmitter, OnChanges, SimpleChanges,
  ChangeDetectorRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { LcCrossService } from '../../core/services/lc-cross.service';
import {
  LcCrossValidationResponseDTO,
  LcCategoryResult,
  LcCrossDocPair,
  LcRuleResult,
  LcCrossExplanation,
  LcWaterfallStep,
  LcXaiExplanation,
  LcProbabilistic,
  LcCausal,
  LcCausalRootCause,
  LcCounterfactual,
  LcSimulationStep,
  LcDecisionConfidence,
} from '../../core/models/lc-cross-validation.model';

type LcTab = 'overview' | 'xai' | 'categories' | 'cross-doc' | 'rules' | 'fuzzy' | 'semantic' | 'conditions' | 'explanations';

const RISK_COLORS: Record<string, { bg: string; fg: string }> = {
  low:      { bg: '#DCFCE7', fg: '#166534' },
  medium:   { bg: '#FEF3C7', fg: '#92400E' },
  high:     { bg: '#FEE2E2', fg: '#991B1B' },
  critical: { bg: '#FEE2E2', fg: '#DC2626' },
  unknown:  { bg: '#F1F5F9', fg: '#64748B' },
};

const LAYER_LABELS: Record<string, string> = {
  // legacy v2 keys
  rules:       'Règles UCP 600',
  fuzzy:       'Entités',
  semantic:    'Sémantique 45A',
  conditions:  'Conditions 47A',
  // v3.0 category keys
  financial:   'Financier',
  goods:       'Marchandises',
  documentary: 'Documentaire',
  shipment:    'Transport',
  parties:     'Parties',
};

const LAYER_COLORS: Record<string, string> = {
  rules:       '#F59E0B',
  fuzzy:       '#1A3A6E',
  semantic:    '#2C5FA3',
  conditions:  '#EC4899',
  financial:   '#16A34A',
  goods:       '#2C5FA3',
  documentary: '#EC4899',
  shipment:    '#F59E0B',
  parties:     '#1A3A6E',
};

const LAYER_WEIGHTS: Record<string, number> = {
  rules: 40, fuzzy: 20, semantic: 25, conditions: 15,
  financial: 35, goods: 25, documentary: 20, shipment: 10, parties: 10,
};

const CATEGORY_ICONS: Record<string, string> = {
  financial: '💰', goods: '📦', documentary: '📋', shipment: '🚢', parties: '👥',
};

@Component({
  selector: 'app-lc-cross-validation',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './lc-cross-validation.component.html',
  styleUrls: ['./lc-cross-validation.component.scss'],
})
export class LcCrossValidationComponent implements OnChanges {
  @Input()  swiftAnalysisId!: number;
  @Input()  commercialValidationId!: number;
  @Output() closed = new EventEmitter<void>();

  loading  = false;
  error    = '';
  result: LcCrossValidationResponseDTO | null = null;
  activeTab: LcTab = 'overview';

  constructor(
    private readonly lcCrossService: LcCrossService,
    private readonly cdr: ChangeDetectorRef,
  ) {}

  ngOnChanges(changes: SimpleChanges): void {
    if ((changes['swiftAnalysisId'] || changes['commercialValidationId']) &&
        this.swiftAnalysisId && this.commercialValidationId) {
      this.run();
    }
  }

  run(): void {
    this.loading = true;
    this.error   = '';
    this.result  = null;
    this.activeTab = 'overview';

    this.lcCrossService.validate(this.swiftAnalysisId, this.commercialValidationId)
      .subscribe({
        next: res => {
          this.result  = res;
          this.loading = false;
          this.cdr.detectChanges();
        },
        error: err => {
          this.error   = err?.error?.message || 'Erreur lors de la validation LC-Commercial.';
          this.loading = false;
          this.cdr.detectChanges();
        },
      });
  }

  // ── Risk / score helpers ─────────────────────────────────────────────

  riskStyle(level: string | null | undefined): { bg: string; fg: string } {
    const k = (level || 'unknown').toLowerCase();
    return RISK_COLORS[k] || RISK_COLORS['unknown'];
  }

  riskIcon(level: string | null | undefined): string {
    const k = (level || '').toLowerCase();
    if (k === 'low')      return '🟢';
    if (k === 'medium')   return '🟡';
    if (k === 'high')     return '🔴';
    if (k === 'critical') return '🚨';
    return '⚪';
  }

  scorePercent(v: number | null | undefined): string {
    if (v == null) return '—';
    return `${Math.round(v * 100)}%`;
  }

  scoreWidth(v: number | null | undefined): number {
    return Math.round((v ?? 0) * 100);
  }

  scoreColor(v: number | null | undefined): string {
    const n = v ?? 0;
    if (n >= 0.80) return '#16A34A';
    if (n >= 0.65) return '#F59E0B';
    return '#DC2626';
  }

  // ── Decision helpers (v3.0 LC-Commercial + v4 UCP) ───────────────────

  decisionStyle(decision: string | null | undefined): { bg: string; fg: string; border: string } {
    const d = (decision || '').toUpperCase();
    if (d === 'ACCEPT')            return { bg: '#DCFCE733', fg: '#15803D',  border: '#16A34A' };
    if (d === 'ACCEPT_WITH_FLAGS') return { bg: '#DCFCE733', fg: '#166534',  border: '#059669' };
    if (d === 'REVIEW')            return { bg: '#E8F0FE33', fg: '#2C5FA3',  border: '#1A3A6E' };
    if (d === 'CONDITIONAL_HOLD')  return { bg: '#FEF3C733', fg: '#92400E',  border: '#F59E0B' };
    if (d === 'DISCREPANT')        return { bg: '#FEF3C733', fg: '#92400E',  border: '#F59E0B' };
    if (d === 'REJECT')            return { bg: '#FEE2E233', fg: '#991B1B',  border: '#DC2626' };
    return { bg: '#F1F5F933', fg: '#64748B', border: '#E2E8F0' };
  }

  decisionIcon(decision: string | null | undefined): string {
    const d = (decision || '').toUpperCase();
    if (d === 'ACCEPT')            return '✅';
    if (d === 'ACCEPT_WITH_FLAGS') return '🟡';
    if (d === 'REVIEW')            return '🔍';
    if (d === 'CONDITIONAL_HOLD')  return '⏸️';
    if (d === 'DISCREPANT')        return '⚠️';
    if (d === 'REJECT')            return '❌';
    return '⏳';
  }

  decisionLabel(decision: string | null | undefined): string {
    const d = (decision || '').toUpperCase();
    if (d === 'ACCEPT')            return 'Accepté';
    if (d === 'ACCEPT_WITH_FLAGS') return 'Accepté avec réserves';
    if (d === 'REVIEW')            return 'À réviser';
    if (d === 'CONDITIONAL_HOLD')  return 'Suspension conditionnelle';
    if (d === 'DISCREPANT')        return 'Discordant';
    if (d === 'REJECT')            return 'Rejeté';
    return decision || '—';
  }

  // ── UCP finding helpers (v4) ─────────────────────────────────────────

  impactLevelStyle(level: string | null | undefined): string {
    const l = (level || '').toUpperCase();
    if (l === 'HIGH')   return 'background:#FEE2E2;color:#991B1B';
    if (l === 'MEDIUM') return 'background:#FEF3C7;color:#92400E';
    if (l === 'LOW')    return 'background:#DCFCE7;color:#166534';
    return 'background:#F1F5F9;color:#64748B';
  }

  findingNatureIcon(nature: string | null | undefined): string {
    const n = (nature || '').toLowerCase();
    if (n === 'true_violation')          return '🚫';
    if (n === 'contractual_modification') return '📝';
    if (n === 'operational_discrepancy')  return '⚠️';
    return 'ℹ️';
  }

  clauseTypeIcon(type: string | null | undefined): string {
    const t = (type || '').toLowerCase();
    if (t === 'financial')       return '💰';
    if (t === 'transport')       return '🚢';
    if (t === 'authentication')  return '🔐';
    if (t === 'identity')        return '🪪';
    if (t === 'compliance')      return '⚖️';
    if (t === 'document_format') return '📄';
    return '📋';
  }

  // ── Status / severity helpers ────────────────────────────────────────

  statusIcon(status: string): string {
    if (status === 'pass')    return '✅';
    if (status === 'fail')    return '❌';
    if (status === 'warning') return '⚠️';
    if (status === 'skipped') return '⏭️';
    return '⏭️';
  }

  statusBorder(status: string): string {
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

  // ── Category helpers (v3.0) ──────────────────────────────────────────

  categoryIcon(k: string): string {
    return CATEGORY_ICONS[k] || '📊';
  }

  categoryStatusIcon(status: string): string {
    if (status === 'pass')    return '✅';
    if (status === 'partial') return '⚠️';
    if (status === 'fail')    return '❌';
    return '⏭️';
  }

  categoryStatusBg(status: string): string {
    if (status === 'pass')    return '#DCFCE722';
    if (status === 'partial') return '#FEF3C722';
    if (status === 'fail')    return '#FEE2E222';
    return '#F1F5F922';
  }

  categoryStatusBorder(status: string): string {
    if (status === 'pass')    return '#16A34A';
    if (status === 'partial') return '#F59E0B';
    if (status === 'fail')    return '#DC2626';
    return '#E2E8F0';
  }

  // ── Explanation helpers ──────────────────────────────────────────────

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

  // ── Layer / breakdown helpers ────────────────────────────────────────

  layerLabel(k: string): string {
    return LAYER_LABELS[k] || k;
  }

  layerColor(k: string): string {
    return LAYER_COLORS[k] || '#6B7280';
  }

  layerWeight(k: string): string {
    const w = LAYER_WEIGHTS[k];
    return w != null ? `${w}%` : '';
  }

  // ── Semantic helpers ─────────────────────────────────────────────────

  semanticLabel(label: string | undefined): string {
    if (!label) return '—';
    if (label === 'MATCH')    return '✅ Concordant';
    if (label === 'PARTIAL')  return '⚠️ Partiel';
    if (label === 'MISMATCH') return '❌ Divergent';
    return label;
  }

  // ── Doc type helper ──────────────────────────────────────────────────

  docTypeLabel(dt: string): string {
    const m: Record<string, string> = {
      bl: 'Bill of Lading', invoice: 'Invoice',
      packing_list: 'Packing List', all: 'Tous',
    };
    return m[dt] || dt?.toUpperCase() || '—';
  }

  // ── Getters ──────────────────────────────────────────────────────────

  get breakdownEntries(): [string, number][] {
    const bd = (this.result?.scoreBreakdown ?? {}) as Record<string, number>;
    const v3Order = ['financial', 'goods', 'documentary', 'shipment', 'parties'];
    const v2Order = ['rules', 'fuzzy', 'semantic', 'conditions'];
    const order = v3Order.some(k => bd[k] != null) ? v3Order : v2Order;
    return order.filter(k => bd[k] != null).map(k => [k, bd[k]] as [string, number]);
  }

  get categoriesEntries(): [string, LcCategoryResult][] {
    const cats = (this.result?.categories ?? {}) as Record<string, LcCategoryResult>;
    const order = ['financial', 'goods', 'documentary', 'shipment', 'parties'];
    return order.filter(k => cats[k] != null).map(k => [k, cats[k]] as [string, LcCategoryResult]);
  }

  get crossDocPairs(): { key: string; label: string; pair: LcCrossDocPair }[] {
    const cd = this.result?.crossDocument;
    if (!cd) return [];
    const pairs: { key: string; label: string; pair: LcCrossDocPair }[] = [];
    if (cd.inv_bl) pairs.push({ key: 'inv_bl', label: 'Invoice ↔ Bill of Lading', pair: cd.inv_bl });
    if (cd.inv_pl) pairs.push({ key: 'inv_pl', label: 'Invoice ↔ Packing List',   pair: cd.inv_pl });
    if (cd.bl_pl)  pairs.push({ key: 'bl_pl',  label: 'B/L ↔ Packing List',       pair: cd.bl_pl  });
    return pairs;
  }

  get negativeExplanations(): LcCrossExplanation[] {
    return (this.result?.explanations ?? []).filter(e => e.impact === 'negative');
  }

  get warningExplanations(): LcCrossExplanation[] {
    return (this.result?.explanations ?? []).filter(e => e.impact === 'warning');
  }

  get positiveExplanations(): LcCrossExplanation[] {
    return (this.result?.explanations ?? []).filter(e => e.impact === 'positive');
  }

  countByStatus(items: LcRuleResult[], status: string): number {
    return (items ?? []).filter(r => r.status === status).length;
  }

  fuzzyPassCount(): number {
    return (this.result?.fuzzyResults ?? []).filter(f => f.status === 'pass').length;
  }

  // ── XAI helpers (v3.0 Shapley) ──────────────────────────────────────

  get xaiExplanations(): LcXaiExplanation[] {
    return (this.result?.explanations ?? []) as unknown as LcXaiExplanation[];
  }

  get xaiNegative(): LcXaiExplanation[] {
    return this.xaiExplanations.filter(e => e.impact === 'negative');
  }

  get xaiWarning(): LcXaiExplanation[] {
    return this.xaiExplanations.filter(e => e.impact === 'warning' && e.type !== 'cross_document');
  }

  get xaiPositive(): LcXaiExplanation[] {
    return this.xaiExplanations.filter(e => e.impact === 'positive');
  }

  get waterfallSteps(): LcWaterfallStep[] {
    return this.result?.xai?.waterfall ?? [];
  }

  waterfallBarWidth(step: LcWaterfallStep): number {
    return Math.min(100, Math.abs(step.delta) * 100 * 5);  // scaled for visibility
  }

  waterfallColor(impactClass: string): string {
    const c: Record<string, string> = {
      critical_negative: '#DC2626',
      high_negative:     '#F97316',
      low_negative:      '#B45309',
      positive:          '#16A34A',
      baseline:          '#2C5FA3',
      final:             '#1A3A6E',
      neutral:           '#E2E8F0',
    };
    return c[impactClass] || '#E2E8F0';
  }

  shapleyBar(phi: number): { width: number; color: string } {
    const pct = Math.min(100, Math.abs(phi) * 100 * 4);
    return {
      width: pct,
      color: phi < 0 ? '#DC2626' : '#16A34A',
    };
  }

  phiLabel(phi: number): string {
    if (phi === 0) return '0.000';
    const sign = phi < 0 ? '−' : '+';
    return `${sign}${Math.abs(phi).toFixed(3)}`;
  }

  trackByIdx = (i: number): number => i;

  // ── Level 4A — Probabilistic getters ─────────────────────────────────

  get probabilistic(): LcProbabilistic | null {
    return (this.result?.xai?.probabilistic) ?? null;
  }

  probaWidth(p: number): number {
    return Math.round(p * 100);
  }

  probaColor(decision: string): string {
    if (decision === 'ACCEPT')            return '#16A34A';
    if (decision === 'ACCEPT_WITH_FLAGS') return '#059669';
    if (decision === 'REVIEW')            return '#1A3A6E';
    if (decision === 'CONDITIONAL_HOLD')  return '#F59E0B';
    if (decision === 'DISCREPANT')        return '#F59E0B';
    if (decision === 'REJECT')            return '#DC2626';
    return '#6B7280';
  }

  ciBarLeft(lower: number): number  { return Math.round(lower * 100); }
  ciBarWidth(lower: number, upper: number): number {
    return Math.round((upper - lower) * 100);
  }

  uncertaintyLabel(std: number): string {
    if (std < 0.03) return 'Faible';
    if (std < 0.07) return 'Modérée';
    return 'Élevée';
  }

  // ── Level 4B — Causal graph getters ──────────────────────────────────

  get causal(): LcCausal | null {
    return (this.result?.xai?.causal) ?? null;
  }

  causalChainLabel(rc: LcCausalRootCause): string {
    if (!rc.causal_children.length) return '';
    return rc.causal_children
      .map(c => c.split('.').pop()?.replace(/lc_/g, '').replace(/_/g, ' ') ?? c)
      .join(' → ');
  }

  interventionGainColor(gain: number): string {
    if (gain >= 0.20) return '#16A34A';
    if (gain >= 0.10) return '#2C5FA3';
    if (gain >= 0.05) return '#F59E0B';
    return '#64748B';
  }

  interventionGainWidth(gain: number): number {
    return Math.min(100, Math.round(gain * 200));
  }

  ruleReadable(rule: string): string {
    return rule.replace(/^lc_/, '').replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  }

  factorRuleLabel(factorId: string): string {
    const part = factorId.includes('.') ? factorId.split('.')[1] : factorId;
    return this.ruleReadable(part);
  }

  isDerived(factorId: string): boolean {
    return (this.result?.xai?.causal?.derived_failures ?? []).includes(factorId);
  }

  // ── Level 4C — Counterfactual getters ────────────────────────────────

  get counterfactuals(): LcCounterfactual[] {
    return this.result?.xai?.counterfactuals ?? [];
  }

  cfScoreColor(score: number): string {
    return this.scoreColor(score);
  }

  cfGainBar(gain: number): number {
    return Math.min(100, Math.round(gain * 300));
  }

  // ── Level 4D — Causal narrative ───────────────────────────────────

  get causalNarrative(): string {
    return this.result?.xai?.causal_narrative ?? '';
  }

  // ── Level 4E — Sequential simulation ─────────────────────────────

  get simulationSteps(): LcSimulationStep[] {
    return this.result?.xai?.simulation ?? [];
  }

  // ── Level 4F — Decision confidence ───────────────────────────────

  get decisionConfidence(): LcDecisionConfidence | null {
    return this.result?.xai?.decision_confidence ?? null;
  }
}
