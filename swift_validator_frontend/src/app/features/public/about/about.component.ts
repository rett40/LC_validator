import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ButtonModule } from 'primeng/button';
import { TagModule } from 'primeng/tag';

@Component({
  selector: 'app-about',
  standalone: true,
  imports: [RouterLink, ButtonModule, TagModule],
  templateUrl: './about.component.html',
  styleUrl: './about.component.scss'
})
export class AboutComponent {
  techs = [
    { name: 'Python 3.12 + spaCy',  cat: 'IA / NLP' },
    { name: 'SWIFT MT700 Parser',    cat: 'SWIFT' },
    { name: 'UCP 600 / ISBP 745',   cat: 'Réglementation' },
    { name: 'Bayesian Scoring',      cat: 'Scoring' },
    { name: 'Angular 21',            cat: 'Frontend' },
    { name: 'PrimeNG 21',            cat: 'UI' },
    { name: 'Spring Boot 3',         cat: 'Backend' },
    { name: 'JWT Auth',              cat: 'Sécurité' },
    { name: 'PostgreSQL',            cat: 'Base de données' },
  ];

  team = [
    { name: 'Équipe IA', role: 'Pipeline NLP, analyse sémantique, scoring bayésien', icon: 'pi-brain' },
    { name: 'Équipe Backend', role: 'API REST Spring Boot, authentification JWT, persistence', icon: 'pi-server' },
    { name: 'Équipe Frontend', role: 'Interface Angular, visualisation, espace admin', icon: 'pi-desktop' },
  ];
}
