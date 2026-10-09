import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { CardModule } from 'primeng/card';
import { TagModule } from 'primeng/tag';
import { SkeletonModule } from 'primeng/skeleton';
import { ChartModule } from 'primeng/chart';
import { ButtonModule } from 'primeng/button';
import { AdminService, DashboardStats } from '../../../core/services/admin.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, DecimalPipe, CardModule, TagModule, SkeletonModule, ChartModule, ButtonModule],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss'
})
export class DashboardComponent implements OnInit {
  loading = true;
  error   = false;
  stats: DashboardStats | null = null;

  donutData: any;
  donutOptions: any;
  barData: any;
  barOptions: any;

  constructor(private admin: AdminService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.reload();
  }

  reload(): void {
    this.loading = true;
    this.error   = false;
    this.admin.getStats().subscribe({
      next: s => {
        this.stats   = s;
        this.loading = false;
        try { this.buildCharts(s); } catch (e) { console.warn('Chart init error:', e); }
        this.cdr.detectChanges();
      },
      error: () => {
        this.loading = false;
        this.error   = true;
        this.cdr.detectChanges();
      }
    });
  }

  private buildCharts(s: DashboardStats): void {
    this.barData = {
      labels: s.weeklyTrend.map(t => t.day),
      datasets: [{
        label: 'Analyses',
        data:  s.weeklyTrend.map(t => t.count),
        backgroundColor: 'rgba(26,58,110,.7)',
        borderRadius: 6,
        borderSkipped: false,
      }]
    };
    this.barOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: '#f1f5f9' }, ticks: { stepSize: 10 }, beginAtZero: true }
      }
    };
    this.donutData = {
      labels: s.decisionBreakdown.map(d => d.label),
      datasets: [{
        data:            s.decisionBreakdown.map(d => d.value),
        backgroundColor: s.decisionBreakdown.map(d => d.color),
        borderWidth: 2,
        borderColor: '#ffffff',
        hoverBorderWidth: 3,
      }]
    };
    this.donutOptions = {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 16, font: { size: 13 } } },
      }
    };
  }

  get acceptPct(): number {
    return this.stats ? Math.round(this.stats.acceptCount / this.stats.totalAnalyses * 100) : 0;
  }
  get reviewPct(): number {
    return this.stats ? Math.round(this.stats.reviewCount / this.stats.totalAnalyses * 100) : 0;
  }
  get rejectPct(): number {
    return this.stats ? Math.round(this.stats.rejectCount / this.stats.totalAnalyses * 100) : 0;
  }
}
