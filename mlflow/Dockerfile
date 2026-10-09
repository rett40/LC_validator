# Image MLflow minimale : serveur de tracking + registre de modèles.
# boto3 requis pour stocker les artefacts dans MinIO (API S3).
FROM python:3.11-slim-bookworm

RUN pip install --no-cache-dir mlflow==2.17.2 boto3==1.35.36

EXPOSE 5000

# Backend SQLite sur volume dédié (aucune dépendance à la base métier),
# artefacts dans le bucket MinIO 'mlflow' (séparé des buckets applicatifs).
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "s3://mlflow", \
     "--serve-artifacts"]
