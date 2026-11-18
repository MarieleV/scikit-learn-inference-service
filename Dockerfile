# ---------- Estágio 1: treina o modelo ----------
FROM python:3.13-slim AS trainer

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt \
    && cp -r /install/. /usr/local/

COPY training/ training/
RUN python -m training.train --output-dir /build/models

# ---------- Estágio 2: imagem final, só com o necessário para servir ----------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

WORKDIR /app

COPY --from=trainer /install /usr/local
COPY app/ app/
COPY --from=trainer /build/models/ models/

# Executa como usuário sem privilégios
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
