import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { TagModule } from 'primeng/tag';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [RouterLink, ButtonModule, CardModule, TagModule],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent {
  features = [
    { icon: 'pi-file-check',  title: 'Conformité du crédit (UCP 600)', desc: 'Vérification automatique des règles UCP 600 et ISBP 745 sur chaque champ du crédit documentaire MT700.' },
    { icon: 'pi-sitemap',     title: 'Cohérence des documents commerciaux', desc: 'Recoupement automatique de la facture, du connaissement (BL) et de la liste de colisage : marchandises, montants, poids, conteneurs.' },
    { icon: 'pi-verified',    title: 'Concordance documents ↔ crédit', desc: 'Contrôle de la concordance entre les documents commerciaux présentés et les exigences du crédit documentaire.' },
    { icon: 'pi-map-marker',  title: 'Contrôle géographique & sanctions', desc: 'Validation des pays, ports et parties, et vérification des listes de sanctions.' },
    { icon: 'pi-chart-line',  title: 'Score & décision documentée', desc: 'Évaluation multicritère aboutissant à une décision claire : Accepté, À vérifier ou Rejeté.' },
    { icon: 'pi-file-pdf',    title: 'Rapports téléchargeables', desc: 'Rapport MT700 et rapport complet (crédit + documents commerciaux) générés en PDF.' },
  ];

  stats = [
    { value: '2', label: 'Niveaux de validation' },
    { value: '<2s',  label: 'Temps d\'analyse' },
    { value: '10+',  label: 'Axes de contrôle' },
    { value: '24/7', label: 'Disponibilité' },
  ];
}
