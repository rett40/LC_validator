# Bookworm (Debian 12) : noms de paquets stables pour WeasyPrint + OCR.
# 'slim' tout court pointe vers Trixie où libgl1-mesa-glx / libgdk-pixbuf2.0-0 ont été renommés.
FROM python:3.11-slim-bookworm

# Dépendances système pour WeasyPrint (pango, cairo, gdk-pixbuf) + OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    fonts-dejavu-core \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirement.txt .
# torch CPU explicite (index PyTorch CPU) — même version que l'env local de référence,
# build CPU léger + résultats alignés. Doit précéder requirement.txt pour éviter le wheel CUDA.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirement.txt

COPY . .

# Utilisateur non-root
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# gunicorn : serveur WSGI de production.
# --workers 1 : les modèles pèsent ~10 GB en RAM — un 2e worker doublerait
#   l'empreinte sans gain (CPU déjà saturé par un forward). La concurrence
#   vient des --threads (I/O, requêtes légères pendant une analyse lourde).
# --preload : warmup exécuté une seule fois dans le master avant le fork.
# --timeout 600 : une analyse multi-documents LayoutLM peut dépasser 30 s.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--preload", \
     "--timeout", "600", "--graceful-timeout", "60", \
     "--bind", "0.0.0.0:5000", "app.api:app"]
