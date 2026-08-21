# ── Etapa 1- Herramientas de Datos (AWS CLI + DVC) ────────────────
FROM python:3.12-slim AS dvc_aws_server

WORKDIR /workspace

RUN pip install --no-cache-dir awscli "dvc[s3]"

# ── Etapa 2- Servidor de MLflow ─────────────────────────────────
FROM python:3.12-slim AS mlflow_server

WORKDIR /mlruns

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlruns/mlflow.db", "--allowed-hosts", "*"]

# ── Etapa 3- Servidor de Entrenamiento de Modelo ─────────────────────────────────
FROM python:3.12-slim AS training_server

ARG APP_VERSION

LABEL maintainer="MLOps Renovacion de Prestamo"
LABEL description="Modelo de entretamiento para predicción de renovacion de Prestamo"
LABEL version=${APP_VERSION}

# No mostrar actualización de pip y evitar escritura de archivos .pyc
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/ /app/requirements/

# ── Instalar dependencias del sistema (mínimas) ───────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r /app/requirements/training.txt

CMD ["tail", "-f", "/dev/null"]


# ── Etapa 4- Servidor de API FastAPI ─────────────────────────────────
FROM python:3.12-slim AS fastapi_server

ARG APP_VERSION
ARG PORT_REMOTE

LABEL maintainer="MLOps Renovacion de Prestamo"
LABEL description="API de predicción para renovacion de Prestamo"
LABEL version=${APP_VERSION}

# No mostrar actualización de pip y evitar escritura de archivos .pyc
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1\
    PORT_REMOTE=${PORT_REMOTE}

WORKDIR /app

COPY requirements/ /app/requirements/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r /app/requirements/fastapi.txt

#EXPOSICION DEL PUERTO DE LA IMAGEN
EXPOSE ${PORT_REMOTE}

# ── Health check para que Docker sepa si el contenedor está sano ──────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx, os; port = os.getenv('PORT_REMOTE'); httpx.get(f'http://localhost:{port}/health')"

#COMANDOS DE EJECUCION DEL APLICATIVO: uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# ── Etapa 5- Gestor de Modelos ML (GCP Bucket + Vertex AI) ────────────────
FROM python:3.12-slim AS gcp_vertexai_server

WORKDIR /workspace

# Instalar dependencias del sistema y Google Cloud SDK CLI
# Sitio oficial: https://docs.cloud.google.com/sdk/docs/install-sdk#deb
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    curl \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update -y && apt-get install google-cloud-cli -y \
    --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*    

CMD ["tail", "-f", "/dev/null"]