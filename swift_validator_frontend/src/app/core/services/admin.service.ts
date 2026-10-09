import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

export interface AgentUser {
  id: number;
  username: string;
  email: string;
  role: string;
  enabled: boolean;
  createdAt: string;
  analysisCount?: number;
}

export interface AgentCreate {
  username: string;
  email: string;
  password: string;
  role: string;
}

export interface DashboardStats {
  totalAgents: number;
  totalAnalyses: number;
  todayAnalyses: number;
  acceptCount: number;
  reviewCount: number;
  rejectCount: number;
  avgScore: number;
  weeklyTrend: { day: string; count: number }[];
  decisionBreakdown: { label: string; value: number; color: string }[];
}

export interface AnalysisHistoryItem {
  id: number;
  lcReference: string;
  decision: string;
  score: number;
  analyzedAt: string;
  warnings: number;
  contradictions: number;
}

@Injectable({ providedIn: 'root' })
export class AdminService {
  private readonly base = `${environment.apiUrl}/api/v1/admin`;

  constructor(private http: HttpClient) {}

  getStats(): Observable<DashboardStats> {
    return this.http.get<DashboardStats>(`${this.base}/stats`, { withCredentials: true });
  }

  getAgents(): Observable<AgentUser[]> {
    return this.http.get<AgentUser[]>(`${this.base}/users`, { withCredentials: true });
  }

  createAgent(data: AgentCreate): Observable<AgentUser> {
    return this.http.post<AgentUser>(`${this.base}/users`, data, { withCredentials: true });
  }

  updateAgent(id: number, data: Partial<AgentUser>): Observable<AgentUser> {
    return this.http.put<AgentUser>(`${this.base}/users/${id}`, data, { withCredentials: true });
  }

  toggleAgent(id: number, enabled: boolean): Observable<AgentUser> {
    return this.http.patch<AgentUser>(`${this.base}/users/${id}/status`, { enabled }, { withCredentials: true });
  }

  deleteAgent(id: number): Observable<void> {
    return this.http.delete<void>(`${this.base}/users/${id}`, { withCredentials: true });
  }

  getAgentHistory(agentId: number): Observable<AnalysisHistoryItem[]> {
    return this.http.get<AnalysisHistoryItem[]>(`${this.base}/users/${agentId}/analyses`, { withCredentials: true });
  }
}
