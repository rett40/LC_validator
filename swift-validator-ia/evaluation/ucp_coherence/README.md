# Évaluation — Taux de détection : incohérences + validation UCP 600

Mesure le **taux de détection (recall)** et la **précision** de :
- **[A]** findings UCP 600 par article (UCP600Validator, mode legacy déterministe)
- **[B]** incohérences tag SWIFT ↔ clause (contradictions Art.5, SwiftSemanticAnalyzer)
- **[C]** cohérence inter-champs (dates)

## Méthode — gold synthétique par injection

On part d'un MT700 **propre** (aucune anomalie). Pour chaque cas, on injecte
**exactement une** anomalie connue (clause, tag ou date). Le label (article UCP
ou champ en contradiction) est donc connu **par construction**. On compare
ensuite ce que le système remonte, après soustraction du bruit de fond du LC
propre.

- **Recall** = cas où l'anomalie injectée est détectée
- **Precision** = TP / (TP + faux positifs hors bruit de fond)

## Périmètre

On évalue la **couche de détection** (règles UCP + contradictions) en lui
fournissant les clauses directement. La segmentation de clauses et le NER ont
leurs propres évaluations (`evaluation/clauses`, `evaluation/swift_ner`) et ne
sont pas réévalués ici.

## Utilisation

```bash
cd swift-validator-ia
python evaluation/ucp_coherence/evaluate_ucp_coherence.py
# figures PNG : nécessitent matplotlib (sinon ignorées proprement)
```

Sorties : `reports/report.json`, `reports/recall_by_category.png`,
`reports/confusion_summary.png`.

## Résultat de référence (2026-06)

Recall **90 %** (18/20), Precision **100 %**. Deux manqués révélés :

1. `transhipment_contradiction_ucp` — l'orthographe `transshipment` (double-s)
   n'est pas couverte par la regex Art.20c legacy (`tranship(?:ment)?`). Le
   moteur de contradictions sémantique, lui, la détecte.
2. `xfield_shipment_after_expiry_dates` — la voie structurée
   `evaluate_cross_field_rules` ne produit rien : elle attend des clés
   `field_a`/`field_b`/`operator`/`code`, alors que
   `ucp_hard_blocklist.json → cross_field_consistency_rules` fournit
   `id`/`rule`/`articles`. La voie texte (`UCP_WARN_SHIPMENT_VS_EXPIRY`) marche.
