import { Routes } from '@angular/router';
import { adminGuard } from './core/guards/admin.guard';
import { agentGuard } from './core/guards/agent.guard';

export const routes: Routes = [

  /* ── Auth ── */
  { path: 'login', loadComponent: () => import('./features/auth/login.component').then(m => m.LoginComponent) },

  /* ── Agent workspace (ADMIN is redirected to /admin/dashboard) ── */
  { path: 'app',        loadComponent: () => import('./component').then(m => m.AppComponent),                canActivate: [agentGuard] },
  { path: 'commercial', loadComponent: () => import('./features/commercial/commercial-docs.component').then(m => m.CommercialDocsComponent), canActivate: [agentGuard] },

  /* ── Admin space ── */
  {
    path: 'admin',
    loadComponent: () => import('./features/admin/admin-shell.component').then(m => m.AdminShellComponent),
    canActivate: [adminGuard],
    children: [
      { path: '',          redirectTo: 'dashboard', pathMatch: 'full' },
      { path: 'dashboard', loadComponent: () => import('./features/admin/dashboard/dashboard.component').then(m => m.DashboardComponent) },
      { path: 'agents',    loadComponent: () => import('./features/admin/agents/agent-list.component').then(m => m.AgentListComponent) },
      { path: 'history',   loadComponent: () => import('./features/admin/history/history.component').then(m => m.HistoryComponent) },
      { path: 'reports',   loadComponent: () => import('./features/admin/reports/reports.component').then(m => m.ReportsComponent) },
    ]
  },

  /* ── Public pages (shell layout) ── */
  {
    path: '',
    loadComponent: () => import('./features/public/public-shell.component').then(m => m.PublicShellComponent),
    children: [
      { path: '',         redirectTo: 'home', pathMatch: 'full' },
      { path: 'home',     loadComponent: () => import('./features/public/home/home.component').then(m => m.HomeComponent) },
      { path: 'services', loadComponent: () => import('./features/public/services-page/services-page.component').then(m => m.ServicesPageComponent) },
      { path: 'about',    loadComponent: () => import('./features/public/about/about.component').then(m => m.AboutComponent) },
      { path: '**',       redirectTo: 'home' },
    ]
  },
];
