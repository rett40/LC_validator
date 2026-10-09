"""
minio_storage.py — Stockage sécurisé des rapports PDF dans MinIO.
Les fichiers sont conservés indéfiniment (Option A).
Les URLs présignées sont régénérées à chaque demande de téléchargement.
"""

import io
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _split_endpoint(raw: str, default_secure: bool) -> tuple[str, bool]:
    """
    Normalise un endpoint MinIO en (host:port, secure).
    Accepte 'host:port', 'http://host:port' ou 'https://host:port'.
    Le SDK MinIO exige le host SANS schéma + un booléen secure séparé.
    """
    raw = raw.strip()
    if raw.startswith("https://"):
        return raw[len("https://"):].rstrip("/"), True
    if raw.startswith("http://"):
        return raw[len("http://"):].rstrip("/"), False
    return raw.rstrip("/"), default_secure


class MinioReportStorage:
    """Client MinIO pour le stockage des rapports PDF.

    Deux endpoints distincts (cf. backend Java) :
      • MINIO_ENDPOINT        — endpoint INTERNE, utilisé pour se connecter et
                                téléverser. Dev local : localhost:9000.
                                Docker : minio:9000 (DNS du réseau swift-net).
      • MINIO_PUBLIC_ENDPOINT — endpoint PUBLIC, utilisé UNIQUEMENT pour signer
                                les URLs présignées remises au navigateur.
                                Doit être joignable depuis le navigateur
                                (localhost:9000 ou le vrai domaine).
    """

    def __init__(self):
        # Import paresseux : le paquet 'minio' n'est pas requis pour démarrer l'app
        # (utile en dev local où il peut être absent). Requis uniquement à l'upload.
        from minio import Minio

        # Défaut localhost:9000 : le dev local fonctionne dès que le conteneur
        # minio expose le port 9000. En Docker, MINIO_ENDPOINT=minio:9000 surcharge.
        default_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        endpoint, secure = _split_endpoint(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"), default_secure
        )

        access_key = os.getenv("MINIO_ACCESS_KEY", "swiftminio")
        secret_key = os.getenv("MINIO_SECRET_KEY", "changeme")

        # Région EXPLICITE : sans elle, le SDK fait un appel réseau
        # get_bucket_location (?location=) sur SON endpoint avant de signer.
        # Pour le client public (localhost:9000), cet appel est impossible
        # depuis le conteneur → la génération de rapport échouait en prod.
        # Avec la région fournie, la présignature est un pur calcul HMAC local.
        region = os.getenv("MINIO_REGION", "us-east-1")

        self.bucket = os.getenv("MINIO_BUCKET_REPORTS", "swift-reports")
        self.client = Minio(endpoint, access_key=access_key,
                            secret_key=secret_key, secure=secure, region=region)

        # Client de signature : pointé sur l'endpoint public pour que l'URL
        # produite soit joignable par le navigateur (l'host signé fait partie
        # de la signature SigV4 — impossible de remplacer l'host après coup).
        public_raw = os.getenv("MINIO_PUBLIC_ENDPOINT")
        if public_raw:
            public_endpoint, public_secure = _split_endpoint(public_raw, secure)
            self.public_client = Minio(
                public_endpoint, access_key=access_key,
                secret_key=secret_key, secure=public_secure, region=region,
            )
        else:
            # Pas d'endpoint public configuré (ex. dev local où interne == public).
            self.public_client = self.client

        self._ensure_bucket()

    def _ensure_bucket(self):
        from minio.error import S3Error
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Bucket '%s' créé.", self.bucket)
        except S3Error as e:
            logger.error("Erreur MinIO bucket: %s", e)

    def upload(self, pdf_bytes: bytes, object_name: str,
               metadata: dict | None = None) -> str:
        """
        Upload un PDF dans MinIO.
        Retourne l'object_name (clé stockée en DB, jamais l'URL directe).
        """
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=io.BytesIO(pdf_bytes),
            length=len(pdf_bytes),
            content_type="application/pdf",
            metadata=metadata or {},
        )
        logger.info("PDF uploadé: %s (%d bytes)", object_name, len(pdf_bytes))
        return object_name

    def presigned_url(self, object_name: str, days: int = 7) -> str:
        """
        Génère une URL présignée valide N jours.
        Le fichier reste dans MinIO indéfiniment (Option A).
        """
        url = self.public_client.presigned_get_object(
            bucket_name=self.bucket,
            object_name=object_name,
            expires=timedelta(days=days),
        )
        return url

    def delete(self, object_name: str):
        """Supprime un objet (si nécessaire manuellement)."""
        self.client.remove_object(self.bucket, object_name)
        logger.info("PDF supprimé: %s", object_name)

    @staticmethod
    def build_object_name(analysis_id: str, report_type: str) -> str:
        """
        Construit la clé MinIO :
        reports/2026/05/abc123-MT700_REPORT.pdf
        """
        now = datetime.utcnow()
        safe_id = str(analysis_id).replace("/", "_")
        return f"reports/{now.year}/{now.month:02d}/{safe_id}-{report_type}.pdf"


# Singleton
_storage: MinioReportStorage | None = None


def get_storage() -> MinioReportStorage:
    global _storage
    if _storage is None:
        _storage = MinioReportStorage()
    return _storage
