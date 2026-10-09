import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, catchError, throwError, timeout } from 'rxjs';
import { AnalysisResponseDTO, DashboardStatsDTO } from '../models/analysis.model';
import { environment } from '../../../environments/environment';

export interface PageResponse<T> {
  content: T[];
  totalElements: number;
  totalPages: number;
  size: number;
  number: number;
}

@Injectable({ providedIn: 'root' })
export class AnalysisService {
  private readonly base = `${environment.apiUrl}/api/v1`;

  constructor(private readonly http: HttpClient) {}

  upload(file: File): Observable<AnalysisResponseDTO> {
    const form = new FormData();
    form.append('file', file);
    return this.http
      .post<AnalysisResponseDTO>(`${this.base}/analyses/upload`, form, { withCredentials: true })
      .pipe(
        timeout(600000),
        catchError((err) => throwError(() => err))
      );
  }

  getById(id: number): Observable<AnalysisResponseDTO> {
    return this.http.get<AnalysisResponseDTO>(`${this.base}/analyses/${id}`, { withCredentials: true });
  }

  list(page = 0, size = 20, riskLevel?: string, lcNumber?: string): Observable<PageResponse<AnalysisResponseDTO>> {
    let params = new HttpParams().set('page', page).set('size', size);
    if (riskLevel) params = params.set('riskLevel', riskLevel);
    if (lcNumber)  params = params.set('lcNumber', lcNumber);
    return this.http.get<PageResponse<AnalysisResponseDTO>>(`${this.base}/analyses`, { params, withCredentials: true });
  }

  getDashboardStats(): Observable<DashboardStatsDTO> {
    return this.http.get<DashboardStatsDTO>(`${this.base}/dashboard/stats`, { withCredentials: true });
  }
}
