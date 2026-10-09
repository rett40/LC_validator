import { Injectable } from '@angular/core';
import { HttpClient }  from '@angular/common/http';
import { Observable }  from 'rxjs';

export interface ReportMeta {
  id:           number;
  analysisId:   number;
  reportType:   'MT700_REPORT' | 'FULL_REPORT';
  displayName:  string;
  fileSize:     number;
  generatedAt:  string;
  downloadUrl:  string;  // URL présignée MinIO valide 7 jours
}

export type ValidationFilter = 'ALL' | 'MT700_ONLY' | 'FULL_VALIDATION';

@Injectable({ providedIn: 'root' })
export class ReportService {

  private base = '/api/reports';

  constructor(private http: HttpClient) {}

  /** Génère le rapport MT700 (ou en recrée un nouveau). */
  generateMt700(analysisId: number): Observable<ReportMeta> {
    return this.http.post<ReportMeta>(`${this.base}/mt700/${analysisId}`, {});
  }

  /**
   * Génère le rapport complet (MT700 + docs commerciaux + cohérence LC↔MT700).
   * `commercialValidationId` permet d'inclure la cross-validation inter-documents
   * et la cohérence des documents avec le MT700 ; omis ⇒ rapport MT700 enrichi seul.
   */
  generateFull(analysisId: number, commercialValidationId?: number): Observable<ReportMeta> {
    const qs = commercialValidationId != null
      ? `?commercialValidationId=${commercialValidationId}`
      : '';
    return this.http.post<ReportMeta>(`${this.base}/full/${analysisId}${qs}`, {});
  }

  /** Liste les rapports disponibles pour une analyse. */
  listReports(analysisId: number): Observable<ReportMeta[]> {
    return this.http.get<ReportMeta[]>(`${this.base}/${analysisId}`);
  }

  /**
   * Régénère une URL présignée fraîche (7 jours).
   * À appeler si l'URL précédente a expiré.
   */
  refreshUrl(reportId: number): Observable<ReportMeta> {
    return this.http.get<ReportMeta>(`${this.base}/refresh/${reportId}`);
  }

  /** Ouvre le PDF dans un nouvel onglet directement via l'URL présignée. */
  openPdf(report: ReportMeta): void {
    window.open(report.downloadUrl, '_blank', 'noopener,noreferrer');
  }

  labelForType(type: 'MT700_REPORT' | 'FULL_REPORT'): string {
    return type === 'MT700_REPORT' ? 'Rapport MT700' : 'Rapport Complet';
  }
}
