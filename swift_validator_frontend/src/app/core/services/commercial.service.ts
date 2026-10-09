import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, catchError, throwError, timeout } from 'rxjs';
import { CommercialValidationResponseDTO } from '../models/commercial.model';
import { environment } from '../../../environments/environment';
import { PageResponse } from './analysis.service';

@Injectable({ providedIn: 'root' })
export class CommercialService {
  private readonly base = `${environment.apiUrl}/api/v1/commercial`;

  constructor(private readonly http: HttpClient) {}

  validate(
    files: File[],
    lcReference?: string,
    swiftAnalysisId?: number
  ): Observable<CommercialValidationResponseDTO> {
    const form = new FormData();
    files.forEach(f => form.append('files', f));
    if (lcReference) form.append('lcReference', lcReference);
    if (swiftAnalysisId != null) form.append('swiftAnalysisId', String(swiftAnalysisId));

    return this.http
      .post<CommercialValidationResponseDTO>(`${this.base}/validate`, form, {
        withCredentials: true,
      })
      .pipe(
        timeout(600000),
        catchError(err => throwError(() => err))
      );
  }

  getById(id: number): Observable<CommercialValidationResponseDTO> {
    return this.http.get<CommercialValidationResponseDTO>(`${this.base}/${id}`, {
      withCredentials: true,
    });
  }

  list(
    page = 0,
    size = 20,
    riskLevel?: string,
    lcReference?: string,
    swiftAnalysisId?: number
  ): Observable<PageResponse<CommercialValidationResponseDTO>> {
    let params = new HttpParams().set('page', page).set('size', size);
    if (riskLevel) params = params.set('riskLevel', riskLevel);
    if (lcReference) params = params.set('lcReference', lcReference);
    if (swiftAnalysisId != null) params = params.set('swiftAnalysisId', swiftAnalysisId);
    return this.http.get<PageResponse<CommercialValidationResponseDTO>>(this.base, {
      params,
      withCredentials: true,
    });
  }
}
