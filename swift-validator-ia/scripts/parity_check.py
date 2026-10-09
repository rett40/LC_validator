"""
parity_check.py — Test de parité golden : local <-> production (Docker).

Compare deux instances de l'API IA sur :
  1. /health          — versions libs, device, revisions Hub, manifest modeles
  2. /commercial/validate — extraction + cross-validation sur les PDF gold/
  3. /swift/analyze   — analyse semantique MT700 (si un echantillon est fourni)

Critere d'acceptation du deploiement : ZERO drift sur (1), champs identiques
et scores egaux a la tolerance pres sur (2) et (3).

Usage :
  # Local (flask run / python app/api.py)  vs  conteneur (port publie 5001)
  python scripts/parity_check.py --a http://localhost:5000 --b http://localhost:5001
  python scripts/parity_check.py --a ... --b ... --mt700 chemin/vers/mt700.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT.parent / "gold"
SCORE_TOLERANCE = 1e-4   # bruit flottant CPU tolere sur les scores

# Metadonnees non deterministes par nature (chronos, fichiers temporaires) :
# jamais comparees — elles different a CHAQUE requete, meme sur un seul serveur.
IGNORED_KEYS = {"processing_time_s", "step_timings", "source", "source_path"}

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [DRIFT] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


# ── 1. Parite d'environnement via /health ────────────────────────────────

def check_health(url_a: str, url_b: str) -> None:
    print("\n== 1. /health : parite d'environnement ==")
    ha = requests.get(f"{url_a}/health", timeout=60).json()
    hb = requests.get(f"{url_b}/health", timeout=60).json()

    def _norm(v):
        # "2.5.1+cu121" et "2.5.1+cpu" = meme version ; le device d'INFERENCE
        # est compare separement (inference_device) — le suffixe de build non.
        if isinstance(v, str):
            return v.split("+")[0]
        if isinstance(v, dict):
            return {k: _norm(x) for k, x in v.items()}
        return v

    pa, pb = ha.get("parity", {}), hb.get("parity", {})
    for key in ("inference_device", "libs", "hub_models_pinned"):
        if _norm(pa.get(key)) != _norm(pb.get(key)):
            fail(f"parity.{key} differe :\n    A={pa.get(key)}\n    B={pb.get(key)}")
        else:
            ok(f"parity.{key} identique")

    ma, mb = ha.get("model_manifest", {}), hb.get("model_manifest", {})
    la, lb = ma.get("local_models", {}), mb.get("local_models", {})
    if la and lb:
        names = set(la) | set(lb)
        for name in sorted(names):
            fa = (la.get(name) or {}).get("files", {})
            fb = (lb.get(name) or {}).get("files", {})
            sha_a = {f: v.get("sha256") for f, v in fa.items()}
            sha_b = {f: v.get("sha256") for f, v in fb.items()}
            if sha_a != sha_b:
                fail(f"manifest {name}: sha256 differents")
            else:
                ok(f"manifest {name}: sha256 identiques")
    else:
        print("  [warn] manifest absent d'un cote (models.yaml non deploye ?)")


# ── 2. Extraction commerciale sur les PDF gold ───────────────────────────

def _diff_numbers(path: str, a, b) -> None:
    if abs(float(a) - float(b)) > SCORE_TOLERANCE:
        fail(f"{path}: {a} != {b}")


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)


def _diff_tree(path: str, a, b) -> None:
    """Compare recursivement : exact pour str/listes, tolerance pour floats."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORED_KEYS:
                continue
            if k not in a or k not in b:
                fail(f"{path}.{k}: present d'un seul cote")
                continue
            _diff_tree(f"{path}.{k}", a[k], b[k])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            fail(f"{path}: longueurs {len(a)} != {len(b)}")
            return
        # Listes de dicts : comparaison insensible a l'ordre (le contenu fait
        # foi ; l'ordre d'emission peut varier avec le scheduling/hash seed).
        if a and all(isinstance(x, dict) for x in a + b):
            a = sorted(a, key=_canon)
            b = sorted(b, key=_canon)
        for i, (xa, xb) in enumerate(zip(a, b)):
            _diff_tree(f"{path}[{i}]", xa, xb)
    elif isinstance(a, float) or isinstance(b, float):
        try:
            _diff_numbers(path, a, b)
        except (TypeError, ValueError):
            if a != b:
                fail(f"{path}: {a!r} != {b!r}")
    elif a != b:
        fail(f"{path}: {a!r} != {b!r}")


def check_commercial(url_a: str, url_b: str) -> None:
    print("\n== 2. /commercial/validate : PDF gold ==")
    if not GOLD_DIR.exists():
        print(f"  [warn] {GOLD_DIR} introuvable — etape sautee")
        return

    # Groupes BL+INV+PL par suffixe de dossier (…_0451, _0892, _1134)
    pdfs = sorted(GOLD_DIR.glob("*.pdf"))
    groups: dict[str, list[Path]] = {}
    for p in pdfs:
        key = p.stem.rsplit("_", 1)[-1]
        groups.setdefault(key, []).append(p)

    for key, files in sorted(groups.items()):
        print(f"  -- groupe {key} ({', '.join(f.name for f in files)})")
        payload = [("files", (f.name, f.read_bytes(), "application/pdf")) for f in files]
        ra = requests.post(f"{url_a}/commercial/validate", files=payload, timeout=900).json()
        payload = [("files", (f.name, f.read_bytes(), "application/pdf")) for f in files]
        rb = requests.post(f"{url_b}/commercial/validate", files=payload, timeout=900).json()
        before = len(FAILURES)
        _diff_tree(f"commercial[{key}]", ra, rb)
        if len(FAILURES) == before:
            ok(f"groupe {key}: sorties identiques")


# ── 3. Analyse semantique MT700 ──────────────────────────────────────────

def check_swift(url_a: str, url_b: str, mt700_path: Path) -> None:
    print("\n== 3. /swift/analyze : MT700 ==")
    text = mt700_path.read_text(encoding="utf-8", errors="replace")
    ra = requests.post(f"{url_a}/swift/analyze", json={"text": text}, timeout=600).json()
    rb = requests.post(f"{url_b}/swift/analyze", json={"text": text}, timeout=600).json()
    before = len(FAILURES)
    _diff_tree("swift", ra, rb)
    if len(FAILURES) == before:
        ok("analyse MT700 : sorties identiques")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="URL instance A (ex: local http://localhost:5000)")
    ap.add_argument("--b", required=True, help="URL instance B (ex: Docker http://localhost:5001)")
    ap.add_argument("--mt700", type=Path, default=None, help="Fichier texte MT700 optionnel")
    ap.add_argument("--skip-commercial", action="store_true",
                    help="Sauter l'etape 2 (longue : ~1 min/groupe de PDF)")
    args = ap.parse_args()

    check_health(args.a, args.b)
    if not args.skip_commercial:
        check_commercial(args.a, args.b)
    if args.mt700:
        check_swift(args.a, args.b, args.mt700)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"PARITE KO — {len(FAILURES)} drift(s). NE PAS DEPLOYER.")
        return 1
    print("PARITE OK — local et production servent les memes resultats.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
