import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { TableModule } from 'primeng/table';
import { ButtonModule } from 'primeng/button';
import { DialogModule } from 'primeng/dialog';
import { InputTextModule } from 'primeng/inputtext';
import { SelectModule } from 'primeng/select';
import { TagModule } from 'primeng/tag';
import { ToastModule } from 'primeng/toast';
import { ConfirmDialogModule } from 'primeng/confirmdialog';
import { ToggleSwitchModule } from 'primeng/toggleswitch';
import { TooltipModule } from 'primeng/tooltip';
import { MessageService, ConfirmationService } from 'primeng/api';
import { AdminService, AgentUser, AgentCreate } from '../../../core/services/admin.service';

@Component({
  selector: 'app-agent-list',
  standalone: true,
  imports: [
    CommonModule, FormsModule, RouterLink,
    TableModule, ButtonModule, DialogModule,
    InputTextModule, SelectModule, TagModule,
    ToastModule, ConfirmDialogModule, ToggleSwitchModule, TooltipModule
  ],
  providers: [MessageService, ConfirmationService],
  templateUrl: './agent-list.component.html',
  styleUrl: './agent-list.component.scss'
})
export class AgentListComponent implements OnInit {
  agents: AgentUser[] = [];
  loading = true;
  showDialog = false;
  editMode   = false;
  saving     = false;

  form: AgentCreate & { id?: number } = {
    username: '',
    email:    '',
    password: '',
    role:     'AGENT',
  };

  roles = [
    { label: 'Agent Bancaire', value: 'AGENT' },
    { label: 'Administrateur', value: 'ADMIN' },
  ];

  globalFilter = '';

  constructor(
    private admin: AdminService,
    private msg: MessageService,
    private confirm: ConfirmationService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading = true;
    this.admin.getAgents().subscribe({
      next:  agents => { this.agents = agents; this.loading = false; this.cdr.detectChanges(); },
      error: ()     => { this.loading = false; this.cdr.detectChanges(); this.msg.add({ severity: 'error', summary: 'Erreur', detail: 'Impossible de charger les agents.' }); }
    });
  }

  openCreate(): void {
    this.editMode = false;
    this.form = { username: '', email: '', password: '', role: 'AGENT' };
    this.showDialog = true;
  }

  openEdit(agent: AgentUser): void {
    this.editMode = true;
    this.form = { id: agent.id, username: agent.username, email: agent.email, password: '', role: agent.role };
    this.showDialog = true;
  }

  get formValid(): boolean {
    if (!this.form.username || !this.form.email) return false;
    if (!this.editMode && this.form.password.length < 8) return false;
    return true;
  }

  save(): void {
    if (!this.formValid) return;
    this.saving = true;

    if (this.editMode && this.form.id) {
      this.admin.updateAgent(this.form.id, { username: this.form.username, email: this.form.email, role: this.form.role }).subscribe({
        next: () => {
          this.saving = false;
          this.showDialog = false;
          this.msg.add({ severity: 'success', summary: 'Succès', detail: 'Agent mis à jour.' });
          this.load();
        },
        error: (err: any) => {
          this.saving = false;
          const detail = err?.error?.message || 'Mise à jour échouée.';
          this.msg.add({ severity: 'error', summary: 'Erreur', detail });
        }
      });
    } else {
      this.admin.createAgent(this.form).subscribe({
        next: () => {
          this.saving = false;
          this.showDialog = false;
          this.msg.add({ severity: 'success', summary: 'Succès', detail: 'Agent créé avec succès.' });
          this.load();
        },
        error: (err: any) => {
          this.saving = false;
          const detail = err?.error?.message || 'Création échouée. Vérifiez les informations.';
          this.msg.add({ severity: 'error', summary: 'Erreur', detail });
        }
      });
    }
  }

  toggleStatus(agent: AgentUser): void {
    const newStatus = !agent.enabled;
    this.admin.toggleAgent(agent.id, newStatus).subscribe({
      next: () => {
        agent.enabled = newStatus;
        this.msg.add({ severity: 'info', summary: 'Mis à jour', detail: `Agent ${newStatus ? 'activé' : 'désactivé'}.` });
      },
      error: () => {
        agent.enabled = !newStatus;
        this.msg.add({ severity: 'error', summary: 'Erreur', detail: 'Changement de statut échoué.' });
      }
    });
  }

  deleteAgent(agent: AgentUser): void {
    this.confirm.confirm({
      message: `Supprimer l'agent « ${agent.username} » ? Cette action est irréversible.`,
      header:  'Confirmer la suppression',
      icon:    'pi pi-trash',
      acceptLabel: 'Supprimer',
      rejectLabel: 'Annuler',
      acceptButtonStyleClass: 'p-button-danger',
      accept: () => {
        this.admin.deleteAgent(agent.id).subscribe({
          next:  () => { this.agents = this.agents.filter(a => a.id !== agent.id); this.msg.add({ severity: 'success', summary: 'Supprimé', detail: `Agent ${agent.username} supprimé.` }); },
          error: () => { this.msg.add({ severity: 'error', summary: 'Erreur', detail: 'Suppression échouée.' }); }
        });
      }
    });
  }

  roleSeverity(role: string): 'warn' | 'info' | 'secondary' {
    if (role === 'ADMIN')  return 'warn';
    if (role === 'AGENT')  return 'info';
    return 'secondary';
  }

  roleLabel(role: string): string {
    if (role === 'ADMIN') return 'Administrateur';
    if (role === 'AGENT') return 'Agent Bancaire';
    return role;
  }

  formatDate(iso: string): string {
    return new Date(iso).toLocaleDateString('fr-FR', { day: '2-digit', month: 'short', year: 'numeric' });
  }
}
