import {
  Component, Input, OnChanges, SimpleChanges,
  ChangeDetectionStrategy, ChangeDetectorRef
} from '@angular/core';
import { CommonModule, PercentPipe } from '@angular/common';
import { ChartModule } from 'primeng/chart';
import {
  RawRiskMlPayload,
  RawScoreBreakdown,
  RawScoreBreakdownDimension,
  RawMissingFieldItem,
  RawExplanation,
  RawLimeScoreWaterfallItem,
  RawUcpDetail,
  RawUcpDetailViolation,
} from '../../core/models/analysis.model';

interface DimCard {
  key:          string;
  label:        string;
  icon:         string;
  score:        number;
  contribution: number;
  weight:       number;
  tone:         string;
  extras:       { label: string; value: string; tone?: string }[];
}

interface ShapItem {
  feature:           string;
  featureLabel:      string;
  value:             string;
  shapValue:         number;
  absShap:           number;
  direction:         string;
  businessDirection: string;
  isDiscordant:      boolean;
  isSignificant:     boolean; // discordant ET au-dessus du seuil de signification
  colorClass:        string;
  barPct:            number;
}

@Component({
  selector: 'app-risk-breakdown',
  standalone: true,
  imports: [CommonModule, ChartModule, PercentPipe],
  templateUrl: './risk-breakdown.component.html',
  styleUrl:    './risk-breakdown.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RiskBreakdownComponent implements OnChanges {
  @Input() riskMl: RawRiskMlPayload | null = null;

  activeTab: 'overview' | 'explanation' | 'compliance' = 'overview';

  breakdown:  RawScoreBreakdown | null = null;
  dimCards:   DimCard[]                = [];
  hasData     = false;

  radarData: any = null;
  radarOpts: any = null;

  limeWaterfallItems: RawLimeScoreWaterfallItem[] = [];
  limeChartData: any  = null;
  limeChartOpts: any  = null;
  limeScoreBase       = 0;
  hasLimeWaterfall    = false;

  shapChartData:  any       = null;
  shapChartOpts:  any       = null;
  shapTableItems: ShapItem[] = [];
  discordantItems: ShapItem[] = [];
  discordantDisplayItems: ShapItem[] = [];
  discordantMoreCount = 0;
  nDiscordant = 0;

  readonly DISC_THRESHOLD = 0.01;

  ucpDetail:       RawUcpDetail | null        = null;
  blockViolations: RawUcpDetailViolation[]    = [];
  highViolations:  RawUcpDetailViolation[]    = [];
  warnViolations:  RawUcpDetailViolation[]    = [];
  ruleContributions: { rule: string; field: string; severity: string; description: string }[] = [];
  missingCritical: RawMissingFieldItem[]      = [];
  missingHigh:     RawMissingFieldItem[]      = [];
  missingOptional: RawMissingFieldItem[]      = [];
  topDrivingDims:  { key: string; score: number; contribution: number }[] = [];

  readonly probaRows = [
    { key: 'LOW',    label: 'Faible', cls: 'green'  },
    { key: 'MEDIUM', label: 'Moyen',  cls: 'orange' },
    { key: 'HIGH',   label: 'Élevé',  cls: 'red'    },
  ];

  readonly DIM_META: Record<string, { label: string; icon: string }> = {
    completeness:   { label: 'Champs MT700',       icon: '📋' },
    ucp_compliance: { label: 'Compliance UCP 600',  icon: '⚖️' },
    coherence:      { label: 'Cohérence docs',      icon: '🔗' },
    country_risk:   { label: 'Risque pays',         icon: '🌍' },
    temporal:       { label: 'Temporel',             icon: '⏱️' },
    amount:         { label: 'Montant & Tenor',     icon: '💰' },
    ml_model:       { label: 'Modèle ML',           icon: '🤖' },
  };

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['riskMl']) {
      this.rebuild();
      this.cdr.markForCheck();
    }
  }

  private rebuild(): void {
    const ml = this.riskMl;
    if (!ml) { this.hasData = false; return; }

    this.breakdown = ml.score_breakdown ?? null;
    this.ucpDetail = ml.ucp_detail ?? null;
    this.hasData   = !!(this.breakdown || ml.missing_fields_detail?.length || this.ucpDetail);

    const expl = ml.explanation ?? null;

    this.buildDimCards();
    this.buildRadarChart();
    this.buildLimeWaterfall(expl);
    this.buildShapChart(expl);
    this.buildUcpViolations();
    this.buildRuleContributions(expl);
    this.buildTopDrivers();
    this.buildMissingFields(ml.missing_fields_detail ?? []);

    this.nDiscordant = expl?.n_discordant_features ?? 0;
  }

  private buildDimCards(): void {
    const dims = this.breakdown?.dimensions;
    if (!dims) { this.dimCards = []; return; }
    const order = ['completeness','ucp_compliance','coherence','country_risk','temporal','amount','ml_model'];
    this.dimCards = order.map(key => {
      const d     = (dims as Record<string, RawScoreBreakdownDimension>)[key];
      const score = Math.round(d?.score ?? 0);
      return {
        key,
        label:        this.DIM_META[key]?.label ?? key,
        icon:         this.DIM_META[key]?.icon  ?? '•',
        score,
        contribution: Math.round((d?.contribution ?? 0) * 10) / 10,
        weight:       Math.round((d?.weight ?? 0) * 100),
        tone:         this.scoreTone(score),
        extras:       this.buildExtras(key, d ?? {}),
      };
    });
  }

  private buildExtras(key: string, d: RawScoreBreakdownDimension): { label: string; value: string; tone?: string }[] {
    switch (key) {
      case 'completeness':
        return [
          { label: 'Critiques manquants', value: String(d['missing_critical_fields'] ?? 0), tone: (d['missing_critical_fields'] as number) > 0 ? 'red' : 'green' },
          { label: 'Taux présence',       value: `${d['critical_fields_present_rate'] ?? 0}%` },
        ];
      case 'ucp_compliance':
        return [
          { label: 'Erreurs (×1.0)',  value: String(d['errors_count']   ?? 0), tone: (d['errors_count'] as number) > 0 ? 'red' : 'green' },
          { label: 'Avert. (×0.4)',   value: String(d['warnings_count'] ?? 0), tone: (d['warnings_count'] as number) > 0 ? 'orange' : 'green' },
          { label: 'Index pondéré',   value: String(d['weighted_violations'] ?? 0) },
        ];
      case 'coherence':
        return [
          { label: 'Dates',         value: `${d['date_coherence_pct'] ?? 50}%`,     tone: (d['date_coherence_pct'] as number) < 50 ? 'red' : 'green' },
          { label: 'Sémantique',    value: `${d['semantic_coherence_pct'] ?? 50}%` },
          { label: 'Contradiction', value: d['contradiction_detected'] ? 'Oui' : 'Non', tone: d['contradiction_detected'] ? 'red' : 'green' },
        ];
      case 'country_risk':
        return [
          { label: 'Applicant',    value: `${d['applicant_risk'] ?? 0}/100`,  tone: (d['applicant_risk'] as number) >= 75 ? 'red' : '' },
          { label: 'Bénéficiaire', value: `${d['beneficiary_risk'] ?? 0}/100`, tone: (d['beneficiary_risk'] as number) >= 75 ? 'red' : '' },
          { label: 'Sanctionné',   value: d['is_sanctioned'] ? 'OUI' : 'Non', tone: d['is_sanctioned'] ? 'red' : 'green' },
        ];
      case 'temporal':
        return [
          { label: 'Validité',     value: `${d['validity_days'] ?? 0}j` },
          { label: 'Expédition',   value: `${d['days_to_shipment'] ?? 0}j` },
          { label: 'Présentation', value: `${d['presentation_days'] ?? 0}j` },
        ];
      case 'amount':
        return [
          { label: 'Tier montant', value: `T${d['tier'] ?? 3}`,      tone: (d['tier'] as number) >= 4 ? 'orange' : '' },
          { label: 'Clause rouge', value: d['red_clause'] ? 'Oui' : 'Non', tone: d['red_clause'] ? 'red' : 'green' },
          { label: 'Tier banque',  value: `T${d['bank_tier'] ?? 3}`, tone: (d['bank_tier'] as number) >= 4 ? 'orange' : '' },
        ];
      case 'ml_model':
        return [
          { label: 'Label ML', value: String(d['label'] ?? '—'), tone: this.riskColor(String(d['label'] ?? '')) },
        ];
      default:
        return [];
    }
  }

  private buildRadarChart(): void {
    const dims = this.breakdown?.dimensions;
    if (!dims) { this.radarData = null; return; }
    const keys   = ['completeness','ucp_compliance','coherence','country_risk','temporal','amount'];
    const labels = keys.map(k => this.DIM_META[k]?.label ?? k);
    const scores = keys.map(k => Math.round((dims as Record<string, RawScoreBreakdownDimension>)[k]?.score ?? 0));

    this.radarData = {
      labels,
      datasets: [{
        label:                'Score de risque',
        data:                 scores,
        fill:                 true,
        backgroundColor:      'rgba(239,68,68,0.10)',
        borderColor:          'rgba(239,68,68,0.75)',
        borderWidth:          2,
        pointBackgroundColor: 'rgba(239,68,68,0.75)',
        pointRadius:          4,
        pointHoverRadius:     6,
      }],
    };
    this.radarOpts = {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, font: { size: 9 }, backdropColor: 'transparent', color: '#94a3b8' },
          pointLabels: { font: { size: 10, weight: '500' }, color: '#475569' },
          grid: { color: 'rgba(0,0,0,0.07)' },
          angleLines: { color: 'rgba(0,0,0,0.07)' },
        },
      },
    };
  }

  private buildLimeWaterfall(expl: RawExplanation | null | undefined): void {
    const items = (expl?.lime_score_waterfall ?? []) as RawLimeScoreWaterfallItem[];
    this.limeWaterfallItems = items;
    this.limeScoreBase      = expl?.lime_score_base ?? 0;
    this.hasLimeWaterfall   = items.length > 0;
    if (!this.hasLimeWaterfall) { this.limeChartData = null; return; }

    const labels  = items.map(i => i.label);
    const posVals = items.map(i => i.score_contribution > 0 ? i.score_contribution : 0);
    const negVals = items.map(i => i.score_contribution < 0 ? Math.abs(i.score_contribution) : 0);

    this.limeChartData = {
      labels,
      datasets: [
        { label: '↑ Augmente le score', data: posVals, backgroundColor: 'rgba(239,68,68,0.80)', borderRadius: 4, borderSkipped: false },
        { label: '↓ Réduit le score',   data: negVals, backgroundColor: 'rgba(34,197,94,0.80)', borderRadius: 4, borderSkipped: false },
      ],
    };

    this.limeChartOpts = {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' as const, labels: { font: { size: 10 }, padding: 12, usePointStyle: true } },
        tooltip: {
          callbacks: {
            label: (ctx: any) => {
              const it = items[ctx.dataIndex];
              if (!it) return '';
              return [
                ` Contribution : ${it.score_contribution > 0 ? '+' : ''}${it.score_contribution.toFixed(2)} pts`,
                ` Valeur : ${this.formatFeatureValue(it.feature, it.value)}`,
                ` SHAP : ${it.shap_value.toFixed(4)}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { font: { size: 10 } },
          grid: { color: 'rgba(0,0,0,0.05)' },
          title: { display: true, text: `Contribution au score (base ${this.limeScoreBase.toFixed(1)} pts)`, font: { size: 10 }, color: '#94a3b8' },
        },
        y: { ticks: { font: { size: 10 } } },
      },
    };
  }

  private buildShapChart(expl: RawExplanation | null | undefined): void {
    const wf = expl?.shap_waterfall ?? [];
    if (!wf.length) {
      this.shapChartData         = null;
      this.shapTableItems        = [];
      this.discordantItems       = [];
      this.discordantDisplayItems = [];
      this.discordantMoreCount   = 0;
      return;
    }

    const top    = wf.slice(0, 12);
    const maxAbs = Math.max(...top.map(i => Math.abs(i.shap_value ?? 0)), 0.001);

    this.shapTableItems = top.map(i => {
      const isDisc      = !!i.is_discordant;
      const absShap     = Math.abs(i.shap_value ?? 0);
      const isSignificant = isDisc && absShap >= this.DISC_THRESHOLD;
      const dir         = i.direction ?? 'down';
      const bd          = i.business_direction ?? 'neutral';

      // Couleur basée sur business_direction (sémantique métier, pas le signe brut du SHAP)
      // business_direction est correct quelle que soit la classe prédite.
      let colorClass: string;
      if (isSignificant)                colorClass = 'orange';
      else if (bd === 'increases_risk') colorClass = 'red';
      else if (bd === 'reduces_risk')   colorClass = 'green';
      else                              colorClass = 'green';

      return {
        feature:           i.feature ?? '',
        featureLabel:      this.featureLabel(i.feature ?? ''),
        value:             this.formatFeatureValue(i.feature ?? '', i.value),
        shapValue:         i.shap_value ?? 0,
        absShap,
        direction:         dir,
        businessDirection: bd,
        isDiscordant:      isDisc,
        isSignificant,
        colorClass,
        barPct: Math.round((absShap / maxAbs) * 100),
      };
    });

    // Seules les discordances significatives alimentent le banner
    this.discordantItems        = this.shapTableItems.filter(i => i.isSignificant)
                                    .sort((a, b) => b.absShap - a.absShap);
    this.discordantDisplayItems = this.discordantItems.slice(0, 5);
    this.discordantMoreCount    = Math.max(0, this.discordantItems.length - 5);

    // Chart : business_direction pour les couleurs (indépendant de la classe prédite)
    const labels   = top.map(i => this.featureLabel(i.feature ?? ''));
    const posVals  = top.map(i => {
      const isSignif = !!i.is_discordant && Math.abs(i.shap_value ?? 0) >= this.DISC_THRESHOLD;
      const bd = i.business_direction ?? 'neutral';
      return (!isSignif && bd === 'increases_risk') ? Math.abs(i.shap_value ?? 0) : 0;
    });
    const negVals  = top.map(i => {
      const isSignif = !!i.is_discordant && Math.abs(i.shap_value ?? 0) >= this.DISC_THRESHOLD;
      const bd = i.business_direction ?? 'neutral';
      return (!isSignif && (bd === 'reduces_risk' || bd === 'neutral')) ? Math.abs(i.shap_value ?? 0) : 0;
    });
    const discVals = top.map(i =>
      (!!i.is_discordant && Math.abs(i.shap_value ?? 0) >= this.DISC_THRESHOLD)
        ? Math.abs(i.shap_value ?? 0) : 0
    );

    this.shapChartData = {
      labels,
      datasets: [
        { label: 'Augmente le risque',          data: posVals,  backgroundColor: 'rgba(239,68,68,0.80)',  borderRadius: 4, borderSkipped: false },
        { label: 'Réduit le risque',            data: negVals,  backgroundColor: 'rgba(34,197,94,0.80)',  borderRadius: 4, borderSkipped: false },
        { label: 'Discordant (corrélation ML)', data: discVals, backgroundColor: 'rgba(245,158,11,0.85)', borderRadius: 4, borderSkipped: false },
      ],
    };

    this.shapChartOpts = {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' as const, labels: { font: { size: 10 }, padding: 12, usePointStyle: true, pointStyleWidth: 10 } },
        tooltip: {
          callbacks: {
            label: (ctx: any) => {
              const i    = ctx.dataIndex;
              const item = top[i];
              return [
                ` SHAP : ${(ctx.raw as number).toFixed(4)}`,
                ` Valeur : ${this.formatFeatureValue(item?.feature ?? '', item?.value)}`,
                ` Direction : ${item?.business_direction === 'increases_risk' ? '↑ risque' : '↓ risque'}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { font: { size: 10 }, callback: (v: any) => v.toFixed(3) },
          grid:  { color: 'rgba(0,0,0,0.05)' },
          title: { display: true, text: 'Valeur SHAP', font: { size: 10 }, color: '#94a3b8' },
        },
        y: { ticks: { font: { size: 10 } } },
      },
    };
  }

  private buildUcpViolations(): void {
    if (!this.ucpDetail) { this.blockViolations = []; this.highViolations = []; this.warnViolations = []; return; }
    this.blockViolations = (this.ucpDetail.block_violations ?? []) as RawUcpDetailViolation[];
    this.highViolations  = (this.ucpDetail.high_violations  ?? []) as RawUcpDetailViolation[];
    this.warnViolations  = (this.ucpDetail.warn_violations  ?? []) as RawUcpDetailViolation[];
  }

  private buildRuleContributions(expl: RawExplanation | null | undefined): void {
    this.ruleContributions = (expl?.rule_contributions ?? []).map(r => ({
      rule:        String(r.rule        ?? ''),
      field:       String(r.field       ?? ''),
      severity:    String(r.severity    ?? ''),
      description: String(r.description ?? ''),
    }));
  }

  private buildTopDrivers(): void {
    const dims = this.breakdown?.dimensions;
    if (!dims) { this.topDrivingDims = []; return; }
    const order = ['completeness','ucp_compliance','coherence','country_risk','temporal','amount'];
    this.topDrivingDims = order
      .map(k => ({
        key:          k,
        score:        Math.round((dims as Record<string,RawScoreBreakdownDimension>)[k]?.score ?? 0),
        contribution: Math.round(((dims as Record<string,RawScoreBreakdownDimension>)[k]?.contribution ?? 0) * 10) / 10,
      }))
      .sort((a, b) => b.contribution - a.contribution)
      .slice(0, 3);
  }

  private buildMissingFields(items: RawMissingFieldItem[]): void {
    this.missingCritical = items.filter(i => i.criticality === 'CRITICAL');
    this.missingHigh     = items.filter(i => i.criticality === 'HIGH');
    this.missingOptional = items.filter(i => i.criticality === 'OPTIONAL');
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  scoreTone(score: number): string {
    if (score >= 70) return 'high';
    if (score >= 40) return 'medium';
    return 'low';
  }

  riskColor(label: string | undefined | null): string {
    switch ((label ?? '').toUpperCase()) {
      case 'HIGH': case 'CRITICAL': return 'red';
      case 'MEDIUM':                return 'orange';
      case 'LOW':  case 'CLEAN':   return 'green';
      default: return '';
    }
  }

  getProba(key: string): number {
    return (this.riskMl?.risk_proba as Record<string, number> | undefined)?.[key] ?? 0;
  }

  bdLabel(bd: string): string {
    switch (bd) {
      case 'increases_risk': return '↑ risque';
      case 'reduces_risk':   return '↓ risque';
      default:               return 'neutre';
    }
  }

  formatFeatureValue(feature: string, value: number | null | undefined): string {
    if (value === null || value === undefined) return '—';
    if ((feature.endsWith('_score') || feature === 'nlp_score' || feature === 'semantic_score') && value <= 1)
      return `${(value * 100).toFixed(0)}%`;
    if (feature === 'ucp_compliance_score') return `${(value * 100).toFixed(0)}%`;
    if (feature.endsWith('_rate') && value <= 1) return `${(value * 100).toFixed(0)}%`;
    if (feature.includes('_days') || feature === 'days_to_shipment') return `${value}j`;
    if (feature === 'lc_amount')
      return value >= 1_000_000 ? `$${(value / 1_000_000).toFixed(1)}M` : `$${(value / 1_000).toFixed(0)}k`;
    if (['transshipment_allowed','charges_beneficiary','red_clause_flag','contradiction_flag'].includes(feature))
      return value ? 'Oui' : 'Non';
    if (['amount_tier','bank_tier'].includes(feature)) return `T${value}`;
    return value % 1 === 0 ? String(value) : value.toFixed(2);
  }

  featureLabel(key: string): string {
    const m: Record<string, string> = {
      lc_amount:                  'Montant LC',
      country_risk_score:         'Risque pays',
      country_risk_applicant:     'Risque applicant',
      country_risk_beneficiary:   'Risque bénéficiaire',
      external_currency_risk:     'Risque devise',
      incoterm_risk:              'Incoterm',
      ucp_compliance_score:       'Score UCP',
      n_ucp_warnings:             'Total violations UCP',
      ucp_errors_count:           'Erreurs UCP',
      ucp_warnings_count:         'Avert. UCP',
      ucp_warning_rate:           'Densité UCP',
      nlp_score:                  'Qualité NLP',
      semantic_score:             'Cohérence sémantique',
      completeness_score:         'Complétude',
      missing_critical_score:     'Champs critiques',
      coherence_dates:            'Cohérence dates',
      contradiction_flag:         'Contradiction',
      days_to_shipment:           'Délai expédition',
      presentation_days:          'Délai présentation',
      validity_days:              'Durée validité',
      amount_tier:                'Tier montant',
      transshipment_allowed:      'Transbordement',
      charges_beneficiary:        'Frais bénéf.',
      red_clause_flag:            'Clause rouge',
      bank_tier:                  'Tier banque',
      tenor_risk:                 'Risque tenor',
    };
    return m[key] ?? key.replace(/_/g, ' ');
  }

  get finalScore(): number { return Math.round(this.riskMl?.final_risk_score ?? this.riskMl?.risk_score ?? 0); }
  get finalTone():  string { return this.scoreTone(this.finalScore); }
  get finalLabel(): string {
    const s = this.finalScore;
    if (s >= 70) return 'HIGH';
    if (s >= 40) return 'MEDIUM';
    return 'LOW';
  }

  get totalMissing():    number  { return this.missingCritical.length + this.missingHigh.length + this.missingOptional.length; }
  get hasMissing():      boolean { return this.totalMissing > 0; }
  get hasUcpDetail():    boolean { return !!this.ucpDetail; }
  get ucpWeightedIndex(): number { return Math.round((this.ucpDetail?.weighted_violation_index ?? 0) * 100) / 100; }
  get isCalibrated():    boolean { return !!(this.riskMl?.explanation?.['_calibration_applied']); }
  get calibrationReason(): string { return String(this.riskMl?.explanation?.['_calibration_reason'] ?? ''); }
  get hasShap():         boolean { return this.shapTableItems.length > 0; }
  get hasLime():         boolean { return this.hasLimeWaterfall; }

  abs(v: number): number { return Math.abs(v); }

}
