import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { TableModule } from 'primeng/table';
import { ButtonModule } from 'primeng/button';
import { TagModule } from 'primeng/tag';
import { SelectModule } from 'primeng/select';
import { SkeletonModule } from 'primeng/skeleton';
import { ToastModule } from 'primeng/toast';
import { MessageService } from 'primeng/api';
import { AdminService, AgentUser, AnalysisHistoryItem } from '../../../core/services/admin.service';
import { ReportService, ReportMeta } from '../../../core/services/report.service';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [
    CommonModule, FormsModule, RouterLink,
    TableModule, ButtonModule, TagModule,
    SelectModule, SkeletonModule, ToastModule
  ],
  providers: [MessageService],
  templateUrl: './history.component.html',
  styleUrl: './history.component.scss'
})
export class HistoryComponent implements OnInit {
  agents:       AgentUser[] = [];
  history:      AnalysisHistoryItem[] = [];
  selectedAgent: AgentUser | null = null;
  loading       = false;
  agentsLoading = true;

  decisionFilter: string | null = null;
  decisionOptions = [
    { label: 'Tous', value: null },
    { label: 'ACCEPTÉ',  value: 'ACCEPT' },
    { label: 'RÉVISION', value: 'REVIEW' },
    { label: 'REJETÉ',   value: 'REJECT' },
  ];

  /** rapports par analysisId : { [id]: ReportMeta[] } */
  reportsByAnalysis: Record<number, ReportMeta[]> = {};
  reportBusyId: number | null = null;

  constructor(
    private admin: AdminService,
    private route: ActivatedRoute,
    private msg: MessageService,
    private cdr: ChangeDetectorRef,
    private reports: ReportService
  ) {}

  ngOnInit(): void {
    this.admin.getAgents().subscribe({
      next: agents => {
        this.agents = agents;
        this.agentsLoading = false;
        this.cdr.detectChanges();
        const id  = this.route.snapshot.queryParamMap.get('agentId');
        const pre = agents.find(a => a.id === Number(id));
        if (pre) { this.selectedAgent = pre; this.loadHistory(pre); }
      },
      error: () => {
        this.agentsLoading = false;
        this.cdr.detectChanges();
        this.msg.add({ severity: 'error', summary: 'Erreur', detail: 'Impossible de charger la liste des agents.' });
      }
    });
  }

  selectAgent(agent: AgentUser): void {
    this.selectedAgent = agent;
    this.decisionFilter = null;
    this.loadHistory(agent);
  }

  loadHistory(agent: AgentUser): void {
    this.loading = true;
    this.history = [];
    this.reportsByAnalysis = {};
    this.admin.getAgentHistory(agent.id).subscribe({
      next: h  => {
        this.history = h;
        this.loading = false;
        this.cdr.detectChanges();
        // Charger les rapports déjà générés pour chaque analyse
        h.forEach(item => this.reports.listReports(item.id).subscribe({
          next: r => { this.reportsByAnalysis[item.id] = r ?? []; this.cdr.detectChanges(); },
          error: () => {},
        }));
      },
      error: () => { this.loading = false; this.cdr.detectChanges(); this.msg.add({ severity: 'error', summary: 'Erreur', detail: 'Impossible de charger l\'historique.' }); }
    });
  }

  reportsFor(id: number): ReportMeta[] { return this.reportsByAnalysis[id] ?? []; }

  openReport(r: ReportMeta): void { this.reports.openPdf(r); }

  generateReport(id: number, type: 'MT700_REPORT' | 'FULL_REPORT'): void {
    this.reportBusyId = id;
    const obs = type === 'MT700_REPORT'
      ? this.reports.generateMt700(id)
      : this.reports.generateFull(id);
    obs.subscribe({
      next: r => {
        const list = this.reportsByAnalysis[id] ?? [];
        this.reportsByAnalysis[id] = [r, ...list.filter(x => x.reportType !== r.reportType)];
        this.reportBusyId = null;
        this.reports.openPdf(r);
        this.cdr.detectChanges();
      },
      error: () => {
        this.reportBusyId = null;
        this.msg.add({ severity: 'error', summary: 'Erreur', detail: 'Génération du rapport échouée.' });
        this.cdr.detectChanges();
      },
    });
  }

  get filteredHistory(): any[] {
    if (!this.decisionFilter) return this.history;
    return this.history.filter(h => h.decision === this.decisionFilter);
  }

  decisionSeverity(d: string): 'success' | 'warn' | 'danger' | 'info' {
    if (d === 'ACCEPT') return 'success';
    if (d === 'REVIEW') return 'warn';
    if (d === 'REJECT') return 'danger';
    return 'info';
  }

  decisionLabel(d: string): string {
    if (d === 'ACCEPT') return 'ACCEPTÉ';
    if (d === 'REVIEW') return 'RÉVISION';
    if (d === 'REJECT') return 'REJETÉ';
    return d;
  }

  formatDate(iso: string): string {
    return new Date(iso).toLocaleDateString('fr-FR', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  get acceptCount(): number { return this.history.filter(h => h.decision === 'ACCEPT').length; }
  get reviewCount(): number { return this.history.filter(h => h.decision === 'REVIEW').length; }
  get rejectCount(): number { return this.history.filter(h => h.decision === 'REJECT').length; }
  get avgScore(): number {
    if (!this.history.length) return 0;
    return Math.round(this.history.reduce((s, h) => s + h.score, 0) / this.history.length);
  }
}
