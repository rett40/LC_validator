import { Component, OnInit } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive, Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { AuthService } from '../../core/services/auth.service';
import { AuthResponse } from '../../core/models/analysis.model';

@Component({
  selector: 'app-admin-shell',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive, CommonModule, ButtonModule],
  templateUrl: './admin-shell.component.html',
  styleUrl: './admin-shell.component.scss'
})
export class AdminShellComponent implements OnInit {
  user: AuthResponse | null = null;
  sidebarCollapsed = false;

  navItems = [
    { label: 'Tableau de bord', icon: 'pi-chart-bar', route: '/admin/dashboard' },
    { label: 'Gestion agents',  icon: 'pi-users',     route: '/admin/agents'    },
    { label: 'Historique',      icon: 'pi-history',   route: '/admin/history'   },
    { label: 'Rapports',        icon: 'pi-file-pdf',  route: '/admin/reports'   },
  ];

  constructor(
    private auth: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.auth.me().subscribe({ next: u => this.user = u, error: () => {} });
  }

  logout(): void {
    this.auth.logout().subscribe({
      next: () => this.router.navigateByUrl('/login'),
      error: () => this.router.navigateByUrl('/login')
    });
  }

  get initials(): string {
    return this.user?.username?.slice(0, 2).toUpperCase() ?? 'AD';
  }
}
