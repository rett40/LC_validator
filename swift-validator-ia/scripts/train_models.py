"""
train_models.py — Script d'entraînement des modèles ML
========================================================
Lance l'entraînement complet des 2 modèles ML du pipeline NLP v2.

Usage (depuis la racine du projet) :
    python scripts/train_models.py --dataset-dir data/data_stati --output-dir models

Structure de projet détectée automatiquement :
    project_lc/
      scripts/train_models.py     <- lance depuis ici
      services/swift_nlp/         <- package trouvé automatiquement
        swift_clause_classifier.py
        swift_semantic_analyzer.py
        ...

Modeles produits :
    models/clause-clf/   -> SentenceTransformer + LogisticRegression
    models/risk-clf/     -> TF-IDF + GradientBoosting (Pipeline sklearn)

Prerequis :
    pip install sentence-transformers scikit-learn joblib
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# RESOLUTION DU PACKAGE
# ======================================================================

def _find_swift_nlp_dir() -> Path:
    """
    Cherche le dossier contenant swift_clause_classifier.py.
    Gere toutes les structures de projet courantes.
    """
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent

    candidates = [
        script_dir,
        project_root / "services" / "swift_nlp",
        project_root / "services" / "swift_nlp_v2",
        project_root / "swift_nlp_v2",
        project_root / "swift_nlp",
        project_root,
    ]

    for c in candidates:
        if (c / "clauses" / "swift_clause_classifier.py").exists():
            logger.info("Package detecte : %s", c)
            return c
        if (c / "swift_clause_classifier.py").exists():
            logger.info("Package detecte (ancienne plat) : %s", c)
            return c

    for match in sorted(project_root.rglob("swift_clause_classifier.py"))[:1]:
        logger.info("Package trouve (recherche) : %s", match.parent)
        return match.parent

    raise FileNotFoundError(
        "swift_clause_classifier.py introuvable sous " + str(project_root) + "\n"
        "Assurez-vous que le package est dans services/swift_nlp/ (sous-dossier clauses/)."
    )


def _setup_sys_path(pkg_dir: Path) -> None:
    for p in [str(pkg_dir), str(pkg_dir.parent), str(pkg_dir.parent.parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    logger.debug("sys.path (4 premiers) : %s", sys.path[:4])


def _load_module_from_file(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None:
        raise ImportError("Impossible de charger " + str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_trainers(pkg_dir: Path):
    """
    Importe train_clause_classifier et train_risk_model.
    Trois strategies dans l'ordre, s'arrete a la premiere qui reussit.
    """
    try:
        from services.swift_nlp.clauses.swift_clause_classifier import train_clause_classifier
        from services.swift_nlp.semantic.swift_semantic_analyzer import train_risk_model
        logger.debug("Import services.swift_nlp OK")
        return train_clause_classifier, train_risk_model
    except ImportError:
        pass

    try:
        from swift_clause_classifier import train_clause_classifier
        from swift_semantic_analyzer import train_risk_model
        logger.debug("Import direct OK")
        return train_clause_classifier, train_risk_model
    except ImportError:
        pass

    parent_name = pkg_dir.parent.name
    pkg_name    = pkg_dir.name
    for full_pkg in [
        "{}.{}".format(parent_name, pkg_name),
        pkg_name,
        "swift_nlp_v2",
        "swift_nlp",
    ]:
        try:
            m_clf = __import__(
                "{}.clauses.swift_clause_classifier".format(full_pkg),
                fromlist=["train_clause_classifier"]
            )
            m_sem = __import__(
                "{}.semantic.swift_semantic_analyzer".format(full_pkg),
                fromlist=["train_risk_model"]
            )
            logger.debug("Import via '%s' OK", full_pkg)
            return m_clf.train_clause_classifier, m_sem.train_risk_model
        except (ImportError, AttributeError):
            try:
                m_clf = __import__(
                    "{}.swift_clause_classifier".format(full_pkg),
                    fromlist=["train_clause_classifier"]
                )
                m_sem = __import__(
                    "{}.swift_semantic_analyzer".format(full_pkg),
                    fromlist=["train_risk_model"]
                )
                logger.debug("Import via '%s' (plat) OK", full_pkg)
                return m_clf.train_clause_classifier, m_sem.train_risk_model
            except (ImportError, AttributeError):
                continue

    try:
        _dep_specs = [
            ("swift_dataset_loader", pkg_dir / "common" / "swift_dataset_loader.py"),
            ("swift_ner_extractor", pkg_dir / "ner" / "swift_ner_extractor.py"),
        ]
        for dep, dep_path in _dep_specs:
            if dep_path.exists() and dep not in sys.modules:
                _load_module_from_file(dep, dep_path)

        spell_path = pkg_dir.parent / "ocr" / "spell_correction.py"
        if spell_path.exists() and "spell_correction" not in sys.modules:
            _load_module_from_file("spell_correction", spell_path)

        clf_path = pkg_dir / "clauses" / "swift_clause_classifier.py"
        sem_path = pkg_dir / "semantic" / "swift_semantic_analyzer.py"
        if not clf_path.is_file():
            clf_path = pkg_dir / "swift_clause_classifier.py"
        if not sem_path.is_file():
            sem_path = pkg_dir / "swift_semantic_analyzer.py"

        mod_clf = _load_module_from_file("swift_clause_classifier", clf_path)
        mod_sem = _load_module_from_file("swift_semantic_analyzer", sem_path)
        logger.debug("Import via importlib.util OK")
        return mod_clf.train_clause_classifier, mod_sem.train_risk_model
    except Exception as e:
        raise ImportError(
            "Impossible d'importer les fonctions d'entrainement depuis {}\n"
            "Erreur : {}\n\n"
            "Verifiez que swift_clause_classifier.py et swift_semantic_analyzer.py "
            "contiennent les fonctions train_clause_classifier() et train_risk_model() "
            "(versions v2).".format(pkg_dir, e)
        ) from e


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entraine les modeles ML du pipeline NLP SWIFT v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python scripts/train_models.py --dataset-dir data/data_stati --output-dir models
  python scripts/train_models.py --dataset-dir data/data_stati --skip-risk
  python scripts/train_models.py --dataset-dir data/data_stati --skip-clauses
        """,
    )
    parser.add_argument("--dataset-dir",  required=True,
                        help="Repertoire contenant les fichiers JSONL")
    parser.add_argument("--output-dir",   default="./models",
                        help="Repertoire de sortie des modeles (defaut: ./models)")
    parser.add_argument("--skip-clauses", action="store_true",
                        help="Ignorer l'entrainement du classifieur de clauses")
    parser.add_argument("--skip-risk",    action="store_true",
                        help="Ignorer l'entrainement du modele de risque")
    parser.add_argument("--embedding-model",
                        default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Modele SentenceTransformer")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir  = Path(args.output_dir).resolve()

    if not dataset_dir.exists():
        logger.error("Repertoire dataset introuvable : %s", dataset_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dataset : %s", dataset_dir)
    logger.info("Output  : %s", output_dir)

    # TR-01 : corrigé — "clause_classification.jsonl" (le point manquait)
    for fname in ["clause_classification.jsonl", "training_samples.jsonl"]:
        if not (dataset_dir / fname).exists():
            logger.warning("Fichier manquant : %s", dataset_dir / fname)

    try:
        pkg_dir = _find_swift_nlp_dir()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    _setup_sys_path(pkg_dir)

    try:
        train_clause_classifier, train_risk_model = _import_trainers(pkg_dir)
        logger.info("Import OK depuis %s", pkg_dir)
    except ImportError as e:
        logger.error("%s", e)
        sys.exit(1)

    success = True

    if not args.skip_clauses:
        logger.info("=" * 55)
        logger.info("Entrainement du classifieur de clauses...")
        logger.info("  Modele : %s", args.embedding_model)
        try:
            train_clause_classifier(
                dataset_dir=str(dataset_dir),
                output_path=str(output_dir / "clause-clf"),
                model_name=args.embedding_model,
            )
            logger.info("OK Classifieur de clauses -> %s", output_dir / "clause-clf")
        except Exception as exc:
            logger.error("ECHEC classifieur clauses : %s", exc, exc_info=True)
            success = False

    if not args.skip_risk:
        logger.info("=" * 55)
        logger.info("Entrainement du modele de risque...")
        try:
            train_risk_model(
                dataset_dir=str(dataset_dir),
                output_path=str(output_dir / "risk-clf"),
            )
            logger.info("OK Modele de risque -> %s", output_dir / "risk-clf")
        except Exception as exc:
            logger.error("ECHEC modele risque : %s", exc, exc_info=True)
            success = False

    logger.info("=" * 55)
    if success:
        logger.info("Entrainement termine - modeles dans : %s", output_dir)
        logger.info("")
        logger.info("Utilisation :")
        logger.info("  from services.swift_nlp import SwiftNLPService, NLPServiceConfig")
        logger.info("  config = NLPServiceConfig(")
        logger.info("      enable_embeddings = True,")
        logger.info("      clause_model_path = r'%s',", output_dir / "clause-clf")
        logger.info("      risk_model_path   = r'%s',", output_dir / "risk-clf")
        logger.info("  )")
        logger.info("  service = SwiftNLPService(config)")
    else:
        logger.error("Certains modeles n'ont pas pu etre entraines.")
        sys.exit(1)


if __name__ == "__main__":
    main()