import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, catchError, throwError, timeout } from 'rxjs';
import { LcCrossValidationResponseDTO } from '../models/lc-cross-validation.model';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class LcCrossService {
  private readonly base = `${environment.apiUrl}/api/v1/lc-cross-validation`;

  constructor(private readonly http: HttpClient) {}

  validate(
    swiftAnalysisId:       number,
    commercialValidationId: number,
  ): Observable<LcCrossValidationResponseDTO> {
    const params = new HttpParams()
      .set('swiftAnalysisId',       String(swiftAnalysisId))
      .set('commercialValidationId', String(commercialValidationId));

    return this.http
      .post<LcCrossValidationResponseDTO>(this.base, null, {
        params,
        withCredentials: true,
      })
      .pipe(
        timeout(120_000),
        catchError(err => throwError(() => err)),
      );
  }
}
