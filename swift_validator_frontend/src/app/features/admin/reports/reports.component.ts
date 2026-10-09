import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { ToastModule } from 'primeng/toast';
import { SkeletonModule } from 'primeng/skeleton';
import { MessageService } from 'primeng/api';
import { AdminService, AgentUser, AnalysisHistoryItem } from '../../../core/services/admin.service';
import { ReportService, ReportMeta } from '../../../core/services/report.service';

interface ReportRow extends AnalysisHistoryItem {
  agentName: string;
}

@Component({
  selector: 'app-admin-reports',
  standalone: true,
  imports: [CommonModule, FormsModule, ToastModule, SkeletonModule],
  providers: [MessageService],
  templateUrl: './reports.component.html',
  styleUrl: './reports.component.scss'
})
export class ReportsComponent implements OnInit {
  rows: ReportRow[] = [];
  loading = true;

  /** rapports déjà générés par analysisId */
  reportsByAnalysis: Record<number, ReportMeta[]> = {};
  /** clé "analysisId:TYPE" en cours de génération */
  busyKey: string | null = null;

  search = '';
  filter: 'ALL' | 'WITH' | 'WITHOUT' = 'ALL';

  constructor(
    private admin: AdminService,
    private reports: ReportService,
    private msg: MessageService,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.admin.getAgents().subscribe({
      next: agents => this.loadAllHistories(agents),
      error: () => {
        this.loading = false;
        this.cdr.detectChanges();
        this.msg.add({ severity: 'error', summary: 'Erreur', detail: 'Impossible de charger les agents.' });
      }
    });
  }

  private loadAllHistories(agents: AgentUser[]): void {
    if (!agents.length) { this.loading = false; this.cdr.detectChanges(); return; }

    forkJoin(
      agents.map(a =>
        this.admin.getAgentHistory(a.id).pipe(
          catchError(() => of([] as AnalysisHistoryItem[]))
        )
      )
    ).subscribe(results => {
      const rows: ReportRow[] = [];
      results.forEach((history, i) => {
        history.forEach(h => rows.push({ ...h, agentName: agents[i].username }));
      });
      rows.sort((a, b) => new Date(b.analyzedAt).getTime() - new Date(a.analyzedAt).getTime());
      this.rows = rows;
      this.loading = false;
      this.cdr.detectChanges();

      // charge les rapports déjà générés
      rows.forEach(r => this.reports.listReports(r.id).subscribe({
        next: list => { this.reportsByAnalysis[r.id] = list ?? []; this.cdr.detectChanges(); },
        error: () => {},
      }));
    });
  }

  reportsFor(id: number): ReportMeta[] { return this.reportsByAnalysis[id] ?? []; }

  reportOfType(id: number, type: 'MT700_REPORT' | 'FULL_REPORT'): ReportMeta | null {
    return this.reportsFor(id).find(r => r.reportType === type) ?? null;
  }

  /** Ouvre si déjà généré, sinon génère puis ouvre. */
  handleReport(id: number, type: 'MT700_REPORT' | 'FULL_REPORT'): void {
    const existing = this.reportOfType(id, type);
    if (existing) { this.reports.openPdf(existing); return; }

    const key = `${id}:${type}`;
    this.busyKey = key;
    const obs = type === 'MT700_REPORT'
      ? this.reports.generateMt700(id)
      : this.reports.generateFull(id);
    obs.subscribe({
      next: r => {
        const list = this.reportsByAnalysis[id] ?? [];
        this.reportsByAnalysis[id] = [r, ...list.filter(x => x.reportType !== r.reportType)];
        this.busyKey = null;
        this.reports.openPdf(r);
        this.cdr.detectChanges();
      },
      error: () => {
        this.busyKey = null;
        this.cdr.detectChanges();
        this.msg.add({ severity: 'error', summary: 'Erreur', detail: 'Génération du rapport échouée.' });
      },
    });
  }

  isBusy(id: number, type: 'MT700_REPORT' | 'FULL_REPORT'): boolean {
    return this.busyKey === `${id}:${type}`;
  }

  get filteredRows(): ReportRow[] {
    const q = this.search.trim().toLowerCase();
    return this.rows.filter(r => {
      const matchSearch = !q
        || (r.lcReference ?? '').toLowerCase().includes(q)
        || (r.agentName ?? '').toLowerCase().includes(q);
      const count = this.reportsFor(r.id).length;
      const matchFilter =
        this.filter === 'ALL'
        || (this.filter === 'WITH' && count > 0)
        || (this.filter === 'WITHOUT' && count === 0);
      return matchSearch && matchFilter;
    });
  }

  get totalReports(): number {
    return Object.values(this.reportsByAnalysis).reduce((s, list) => s + list.length, 0);
  }

  get dossiersWithReport(): number {
    return Object.values(this.reportsByAnalysis).filter(l => l.length > 0).length;
  }

  decisionLabel(d: string): string {
    if (d === 'ACCEPT') return 'Accepté';
    if (d === 'REVIEW') return 'À vérifier';
    if (d === 'REJECT') return 'Rejeté';
    return d || '—';
  }

  decisionClass(d: string): string {
    if (d === 'ACCEPT') return 'accept';
    if (d === 'REVIEW') return 'review';
    if (d === 'REJECT') return 'reject';
    return 'neutral';
  }

  formatDate(iso: string): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('fr-FR', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  trackById = (_: number, r: ReportRow) => r.id;
}
