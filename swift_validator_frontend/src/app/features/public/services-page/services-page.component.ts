import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { TagModule } from 'primeng/tag';
@Component({
  selector: 'app-services-page',
  standalone: true,
  imports: [RouterLink, ButtonModule, CardModule, TagModule],
  templateUrl: './services-page.component.html',
  styleUrl: './services-page.component.scss'
})
export class ServicesPageComponent {
  services = [
    {
      icon: 'pi-file-check',
      tag: 'Crédit documentaire',
      title: 'Validation du MT700 (UCP 600 / ISBP 745)',
      desc: 'Contrôle automatique de tous les champs du crédit documentaire MT700 selon les règles UCP 600 et ISBP 745 : validité des dates, montants, Incoterms et clauses documentaires.',
      items: ['Contrôle des champs obligatoires du crédit', 'Détection des Incoterms obsolètes (DES, DEQ, DAF, DDU)', 'Cohérence montant / devise / tolérance', 'Contrôle des délais et de la période de présentation']
    },
    {
      icon: 'pi-sitemap',
      tag: 'Documents commerciaux',
      title: 'Cohérence des documents commerciaux',
      desc: 'Recoupement automatique de la facture commerciale, du connaissement (BL) et de la liste de colisage afin de garantir leur cohérence mutuelle avant présentation.',
      items: ['Alignement des marchandises facture ↔ colisage', 'Cohérence des poids brut / net et conteneurs (BL ↔ colisage)', 'Vérification des montants et quantités', 'Concordance des parties (vendeur, expéditeur, destinataire)']
    },
    {
      icon: 'pi-verified',
      tag: 'Concordance',
      title: 'Concordance documents ↔ crédit',
      desc: 'Vérification que les documents commerciaux présentés respectent les exigences du crédit documentaire (marchandises, montant, Incoterm, parties).',
      items: ['Marchandises du crédit ↔ documents', 'Montant et devise présentés ↔ crédit', 'Bénéficiaire ↔ vendeur / expéditeur', 'Incoterm et conditions d\'expédition']
    },
    {
      icon: 'pi-map-marker',
      tag: 'Géographie & sanctions',
      title: 'Contrôle géographique & réglementaire',
      desc: 'Validation des pays, ports et parties avec codes ISO 3166-1, et vérification face aux listes de sanctions.',
      items: ['Résolution des codes pays (FR, TN, DE…)', 'Correspondance bénéficiaire ↔ pays', 'Validation des ports de chargement / déchargement', 'Vérification des listes de sanctions']
    },
    {
      icon: 'pi-chart-bar',
      tag: 'Décision',
      title: 'Score & décision documentée',
      desc: 'Évaluation multicritère aboutissant à une décision claire et traçable : Accepté, À vérifier ou Rejeté, avec justification réglementaire.',
      items: ['Score de conformité du crédit', 'Score de cohérence des documents', 'Décision traçable et auditée', 'Historique et tendances par agent']
    },
    {
      icon: 'pi-file-pdf',
      tag: 'Rapports',
      title: 'Rapports MT700 & complet',
      desc: 'Génération de rapports PDF : rapport du crédit MT700 seul, ou rapport complet incluant la cohérence des documents commerciaux et leur concordance avec le crédit.',
      items: ['Rapport MT700 (crédit documentaire)', 'Rapport complet (crédit + documents)', 'Téléchargement et archivage sécurisés', 'Accès contrôlé par rôle']
    },
  ];

  openFaqIndex: number | null = null;

  toggleFaq(i: number): void {
    this.openFaqIndex = this.openFaqIndex === i ? null : i;
  }

  faqs = [
    { q: 'Quels documents sont pris en charge ?', r: 'Le crédit documentaire SWIFT MT700, ainsi que les principaux documents commerciaux : facture commerciale, connaissement (Bill of Lading) et liste de colisage. L\'extraction fonctionne aussi sur les documents scannés.' },
    { q: 'Quelle est la différence entre les deux niveaux de validation ?', r: 'Le premier niveau vérifie la conformité réglementaire du crédit documentaire (UCP 600). Le second contrôle la cohérence des documents commerciaux entre eux et leur concordance avec les exigences du crédit.' },
    { q: 'Que signifie la décision « À vérifier » ?', r: 'Elle indique qu\'une revue manuelle par un agent bancaire est recommandée : des anomalies non bloquantes ont été détectées et nécessitent un jugement expert.' },
    { q: 'Les données sont-elles conservées ?', r: 'Chaque dossier est archivé avec sa référence, la décision, le score et la date. Les rapports PDF sont stockés de façon sécurisée et l\'accès est contrôlé par rôle (Administrateur / Agent).' },
  ];
}
