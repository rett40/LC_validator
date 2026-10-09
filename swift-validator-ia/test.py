from services.swift_nlp.semantic.swift_semantic_analyzer import SwiftSemanticAnalyzer, SemanticAnalysisResult

# 1. Préparer vos données
entities = {
    "swift_type": "MT700",
    "lc_number": "LC123456",
    "issue_date": "2026-03-01",
    "expiry_date": "2026-03-30",
    "shipment_date": "2026-03-15",
    "amount": "100000",
    "currency": "USD",
    "applicant": "COMPANY A",
    "beneficiary": "COMPANY B",
    "incoterm": "FOB",
    "partial_shipments": "NOT ALLOWED",   # champ :43P:
    "transhipment": "ALLOWED",            # champ :43T:
    "applicable_rules": "UCP 600",
    "documents_required": "COMMERCIAL INVOICE, BILL OF LADING, INSURANCE CERTIFICATE"
}

# Texte brut du message (notamment pour les clauses)
raw_text = """
:43P: NOT ALLOWED
:43T: ALLOWED
:47A:
PARTIAL SHIPMENTS ARE PROHIBITED.
TRANSHIPMENT IS NOT PERMITTED.
:45A:
GOODS: MACHINERY FOB SHANGHAI
"""

# 2. Instancier l'analyseur
analyzer = SwiftSemanticAnalyzer(
    risk_model_path=None,          # chemin vers un modèle ML optionnel
    enable_semantic_coherence=True,
    fuzzy_threshold=82
)

# 3. Lancer l'analyse
result: SemanticAnalysisResult = analyzer.analyze(entities, raw_text)

# 4. Examiner les incohérences
print("Contradictions détectées :")
for c in result.contradictions:
    print(f" - Tag {c.tag_field}:{c.tag_value} vs clause \"{c.clause_text}\" → {c.recommendation}")

print("\nWarnings :")
for w in result.warnings:
    print(f" - {w.severity}: {w.message}")

print(f"\nScore de cohérence : {result.semantic_score}/100")