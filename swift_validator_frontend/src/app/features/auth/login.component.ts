import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';
import { Router } from '@angular/router';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink, ButtonModule, InputTextModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent {
  username = '';
  password = '';
  loading  = false;
  error    = '';
  showPwd  = false;

  constructor(private auth: AuthService, private router: Router) {}

  onSubmit(): void {
    if (!this.username || !this.password || this.loading) return;
    this.loading = true;
    this.error   = '';
    this.auth.login(this.username, this.password).subscribe({
      next: (res) => {
        this.loading = false;
        if (res.role === 'ADMIN') {
          this.router.navigateByUrl('/admin/dashboard');
        } else {
          this.router.navigateByUrl('/app');
        }
      },
      error: (err: HttpErrorResponse) => {
        this.loading = false;
        if (err.status === 401) { this.error = 'Identifiants invalides'; return; }
        if (err.status === 0)   { this.error = 'Backend inaccessible — vérifiez que le service tourne sur le port 8080.'; return; }
        this.error = 'Erreur de connexion. Réessayez.';
      }
    });
  }
}
