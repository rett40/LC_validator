"""
external_validation_b/scheduler.py
Planificateur  –  mise à jour mensuelle (30 jours, configurable)

Répertoire de données : data/scraping/
    data/scraping/latest/   ← toujours la version la plus récente
    data/scraping/archive/  ← archives mensuelles

Usage :
    python scheduler.py --run-now          # Refresh si nécessaire
    python scheduler.py --run-now --force  # Forcer le refresh
    python scheduler.py --status           # État du cache
    python scheduler.py --daemon           # Mode démon (bloque le terminal)

Variables d'environnement :
    SANCTIONS_DATA_DIR   — répertoire racine (défaut: data/scraping)
    CACHE_MAX_AGE_DAYS   — durée de validité en jours (défaut: 30)
    INCLUDE_GLEIF_SNAPSHOT — activer snapshot GLEIF (défaut: 0)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Chemin ancré au module (parents[2] = racine swift-validator-ia), JAMAIS au CWD.
# Surcharge possible via SANCTIONS_DATA_DIR (utilisé en Docker).
_PROJECT_ROOT   = Path(__file__).resolve().parents[2]
_SCRAPING_ROOT  = Path(os.environ.get("SANCTIONS_DATA_DIR", str(_PROJECT_ROOT / "data" / "scraping")))
DATA_DIR        = _SCRAPING_ROOT / "latest"
ARCHIVE_DIR     = _SCRAPING_ROOT / "archive"
REFRESH_DAYS    = int(os.environ.get("CACHE_MAX_AGE_DAYS", "30"))   # mensuel
INCLUDE_GLEIF   = os.environ.get("INCLUDE_GLEIF_SNAPSHOT", "0") == "1"

SCHEDULER_LOG   = _SCRAPING_ROOT / "scheduler.log"
LAST_RUN_FILE   = _SCRAPING_ROOT / ".last_run.json"

for _d in (_SCRAPING_ROOT, DATA_DIR, ARCHIVE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Logging (UTF-8 safe sur Windows et Linux)
# ─────────────────────────────────────────────────────────────────────────────
def _utf8_handler(stream=None) -> logging.StreamHandler:
    import sys as _sys
    s = stream or open(
        _sys.stdout.fileno(), mode="w", encoding="utf-8",
        errors="replace", closefd=False, buffering=1,
    )
    h = logging.StreamHandler(s)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    return h


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(SCHEDULER_LOG, encoding="utf-8"),
        _utf8_handler(),
    ],
)
log = logging.getLogger("scheduler")

# ─────────────────────────────────────────────────────────────────────────────
#  APScheduler (optionnel)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
    _HAS_APSCHEDULER = True
except ImportError:
    _HAS_APSCHEDULER = False
    log.warning("APScheduler non installé — pip install apscheduler\n"
                "Le refresh manuel reste disponible via run_refresh().")


# ─────────────────────────────────────────────────────────────────────────────
#  Imports internes différés
# ─────────────────────────────────────────────────────────────────────────────
def _import_modules():
    """Import différé pour éviter les imports circulaires."""
    try:
        from .scraper import run_scraper
        from .cache_manager import invalidate_all, cache_status
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from scraper import run_scraper
        from cache_manager import invalidate_all, cache_status
    return run_scraper, invalidate_all, cache_status


# ─────────────────────────────────────────────────────────────────────────────
#  Gestion du fichier .last_run.json
# ─────────────────────────────────────────────────────────────────────────────
def _read_last_run() -> dict:
    if LAST_RUN_FILE.exists():
        try:
            with open(LAST_RUN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _write_last_run(stats: dict, success: bool):
    payload = {
        "timestamp":     datetime.utcnow().isoformat() + "Z",
        "success":       success,
        "refresh_every": f"{REFRESH_DAYS} jours",
        "data_dir":      str(_SCRAPING_ROOT),
        "stats":         stats,
    }
    with open(LAST_RUN_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _should_refresh() -> bool:
    """
    Retourne True si :
      - aucun dernier refresh réussi enregistré, ou
      - le dernier refresh date de plus de REFRESH_DAYS jours.
    """
    last = _read_last_run()
    if not last.get("timestamp") or not last.get("success"):
        return True
    try:
        last_dt  = datetime.fromisoformat(last["timestamp"].replace("Z", ""))
        elapsed  = (datetime.utcnow() - last_dt).days
        return elapsed >= REFRESH_DAYS
    except Exception:
        return True


def _current_month_archive_exists() -> bool:
    """Vérifie si l'archive du mois courant est déjà présente."""
    month = datetime.now().strftime("%Y-%m")
    return (ARCHIVE_DIR / f"country_risk_{month}.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
#  Job principal
# ─────────────────────────────────────────────────────────────────────────────
def run_refresh(force: bool = False):
    """
    Lance le scraper si nécessaire (ou forcé).

    Args:
        force: si True, ignore les vérifications de date et force le recalcul.
    """
    run_scraper, invalidate_all, cache_status = _import_modules()

    if not force and not _should_refresh() and _current_month_archive_exists():
        last = _read_last_run()
        log.info(
            "✅ Cache mensuel à jour (dernier refresh : %s) — prochain dans %d jours",
            last.get("timestamp", "inconnu"), REFRESH_DAYS,
        )
        return

    log.info(
        "🔄 Démarrage du refresh mensuel (fréquence : %d jours) [force=%s]",
        REFRESH_DAYS, force,
    )

    try:
        stats = run_scraper(force=force)
        invalidate_all()
        _write_last_run(stats, success=True)

        if stats.get("skipped"):
            log.info("ℹ️  Scraper sauté — données du mois %s déjà présentes", stats.get("month"))
        else:
            log.info(
                "✅ Refresh terminé — %d pays | CRITICAL:%d HIGH:%d MEDIUM:%d LOW:%d",
                stats.get("total_countries", 0),
                stats.get("critical", 0), stats.get("high", 0),
                stats.get("medium", 0), stats.get("low", 0),
            )
            v = stats.get("validation", {})
            if v.get("country_errors") or v.get("currency_errors"):
                log.warning(
                    "⚠️ Validation — pays KO: %d | devises KO: %d | avertissements: %d",
                    v.get("country_errors", 0),
                    v.get("currency_errors", 0),
                    v.get("warnings", 0),
                )

    except Exception as e:
        log.error("❌ Refresh échoué : %s", e, exc_info=True)
        _write_last_run({}, success=False)
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Scheduler APScheduler
# ─────────────────────────────────────────────────────────────────────────────
_scheduler: "BackgroundScheduler | None" = None


def _on_job_event(event):
    if event.exception:
        log.error("❌ Job APScheduler échoué : %s", event.exception)
    else:
        log.info("✅ Job APScheduler exécuté avec succès.")


def start_scheduler() -> bool:
    """
    Démarre le planificateur en arrière-plan.
    - Refresh immédiat si le cache est absent/expiré.
    - Planifie ensuite un refresh toutes les REFRESH_DAYS jours.
    Retourne True si démarré, False si APScheduler est absent.
    """
    global _scheduler

    if not _HAS_APSCHEDULER:
        log.warning("APScheduler non disponible — utiliser run_refresh() manuellement.")
        return False

    if _scheduler and _scheduler.running:
        log.info("Scheduler déjà en cours.")
        return True

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_listener(_on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    _scheduler.add_job(
        run_refresh,
        trigger=IntervalTrigger(days=REFRESH_DAYS),
        id="sanctions_monthly_refresh",
        name=f"Refresh mensuel listes sanctions (/{REFRESH_DAYS}j)",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=86400,   # Tolérance 24h si l'app était éteinte
    )
    _scheduler.start()
    log.info(
        "🗓  Scheduler démarré — refresh tous les %d jours | "
        "données → %s | GLEIF: %s",
        REFRESH_DAYS, _SCRAPING_ROOT,
        "activé" if INCLUDE_GLEIF else "désactivé",
    )

    # Refresh immédiat si nécessaire
    if _should_refresh() or not _current_month_archive_exists():
        log.info("⚡ Cache absent/expiré — refresh immédiat...")
        try:
            run_refresh(force=False)
        except Exception as e:
            log.error("Refresh initial échoué : %s", e)

    return True


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("Scheduler arrêté.")
    _scheduler = None


def get_next_run() -> str | None:
    if not _scheduler or not _scheduler.running:
        return None
    job = _scheduler.get_job("sanctions_monthly_refresh")
    if job and job.next_run_time:
        return job.next_run_time.isoformat()
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  --status
# ─────────────────────────────────────────────────────────────────────────────
def _cli_status():
    try:
        from .cache_manager import cache_status
    except ImportError:
        from cache_manager import cache_status

    print("\n" + "=" * 68)
    print(f"   STATUT DU CACHE  –  répertoire : {_SCRAPING_ROOT}")
    print("=" * 68)

    status = cache_status()

    # Archives mensuelles
    arch = status.pop("_archives", {})

    for key, info in status.items():
        available = info.get("available", False)
        icon = "✅" if available and not info.get("needs_refresh") else (
               "⚠️ " if available else "❌")
        print(f"\n  {icon}  {key.upper():<16} {info.get('path', '')}")
        if available:
            print(f"          Mis à jour : {info.get('updated_at', 'N/A')}")
            print(f"          Âge        : {info.get('age_days', 'N/A')} jours "
                  f"(refresh tous les {info.get('refresh_every', '?')})")
            print(f"          Entrées    : {info.get('count', 'N/A')}")
            print(f"          Refresh    : {'NÉCESSAIRE ⚠️' if info.get('needs_refresh') else 'OK'}")
        else:
            print("          Cache absent — exécuter le scraper")

    print(f"\n  📦 Archives mensuelles ({arch.get('count', 0)}) → {arch.get('dir', '')}")
    if arch.get("months"):
        print("     Mois disponibles : " + ", ".join(arch["months"]))

    last = _read_last_run()
    print("\n" + "-" * 68)
    if last:
        print(f"  Dernier refresh  : {last.get('timestamp', 'N/A')}")
        print(f"  Statut           : {'✅ Succès' if last.get('success') else '❌ Échec'}")
        s = last.get("stats", {})
        if s and not s.get("skipped"):
            print(f"  Pays traités     : {s.get('total_countries', 'N/A')}")
            v = s.get("validation", {})
            print(f"  Validation       : erreurs pays={v.get('country_errors', 0)} "
                  f"devises={v.get('currency_errors', 0)} "
                  f"warnings={v.get('warnings', 0)}")
    else:
        print("  Aucun historique de refresh.")
    print(f"  Prochain refresh : dans {REFRESH_DAYS} jours")
    print("=" * 68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Planificateur de refresh mensuel des listes de sanctions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python scheduler.py --run-now          # Refresh si nécessaire
  python scheduler.py --run-now --force  # Refresh forcé (même si données présentes)
  python scheduler.py --status           # Voir l'état du cache
  python scheduler.py --daemon           # Lancer en mode démon

Données sauvegardées dans :
  data/scraping/latest/   ← fichiers courants
  data/scraping/archive/  ← archives par mois (YYYY-MM)
        """,
    )
    parser.add_argument("--run-now", action="store_true", help="Lancer un refresh maintenant")
    parser.add_argument("--force",   action="store_true", help="Forcer le refresh")
    parser.add_argument("--status",  action="store_true", help="Afficher le statut du cache")
    parser.add_argument("--daemon",  action="store_true", help="Lancer en mode démon")
    args = parser.parse_args()

    if args.status:
        _cli_status()

    if args.run_now:
        log.info("Refresh manuel demandé (force=%s)", args.force)
        run_refresh(force=args.force)

    if args.daemon:
        if not _HAS_APSCHEDULER:
            print("❌ APScheduler requis : pip install apscheduler")
            sys.exit(1)
        log.info("Mode démon — Ctrl+C pour arrêter")
        start_scheduler()
        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            stop_scheduler()
            log.info("Démon arrêté.")

    if not any([args.status, args.run_now, args.daemon]):
        parser.print_help()


if __name__ == "__main__":
    main()