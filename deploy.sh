#!/bin/bash
# deploy.sh — Flujo manual de despliegue desarrollo

set -e  # salir si cualquier comando falla

ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    echo "Cargando configuración local desde $ENV_FILE..."
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "⚠️ Advertencia: No se encontró el archivo $ENV_FILE. Se usarán variables del sistema."
fi

VERSION=${1:-${IMAGE_VERSION:-latest}}
IMAGE_NAME=${IMAGE_NAME:-"renovacion_prestamo-image"}
COMPOSE_FILE="compose.yml"

echo "===================================================================="
echo " DEPLOY DESARROLLO — Entorno: ${ENV:-dev} | Versión: $VERSION"
echo "===================================================================="

# ── PASO 1: Guardar versión actual para rollback ──────────────────────────────
echo "[1/5] Guardando versión actual para rollback..."
CURRENT=$(docker images --format "{{.Tag}}" "$IMAGE_NAME" 2>/dev/null | head -1 || echo "none")
echo "  Versión actual: $CURRENT"

# ── PASO 2: Build de todas las imágenes ──────────────────────────────────────
echo "[2/5] Construyendo imágenes Docker..."
docker compose -f $COMPOSE_FILE build --build-arg APP_VERSION="$APP_VERSION" --build-arg PORT_LOCAL="$PORT_LOCAL" --build-arg PORT_REMOTE="$PORT_REMOTE"
echo "  Build OK"

# ── PASO 3: Levantar el entorno Secuencialmente──────────────────────────────────────
echo "[3/5] Levantando servicios de forma secuencial..."
docker compose -f $COMPOSE_FILE up -d mlflow
echo "  Esperando que el healthcheck de MLflow responda exitosamente..."
until [ "$(docker inspect --format='{{.State.Health.Status}}' "$DOCKER_MLFLOW_NAME")" = "healthy" ]; do
    sleep 2
done

docker compose -f $COMPOSE_FILE up -d fastapi
echo "  Servicios instanciados."

# ── PASO 4: Tag de la versión ─────────────────────────────────────────────────
echo "[4/5] Aplicando etiquetas estables de Docker..."
if docker image inspect "$IMAGE_NAME:$VERSION" >/dev/null 2>&1; then
    docker tag "$IMAGE_NAME:$VERSION" "$IMAGE_NAME:latest"
    echo "  Imagen taggeada exitosamente: $IMAGE_NAME:latest"
else
    echo "  ⚠️ No se encontró la etiqueta estática de versión $VERSION. Saltando retaggeo local."
fi

# ── PASO 5: Verificación final ────────────────────────────────────────────────
echo "[5/5] Verificación final..."
VERIFY_PORT_LOCAL=${PORT_LOCAL:-8085}
VERIFY_PORT_MLFLOW=${MLFLOW_PORT:-5000}
echo "  Enviando petición de prueba al endpoint de salud en el puerto $VERIFY_PORT_LOCAL..."

for i in {1..10}; do
    if curl -sf "http://localhost:$VERIFY_PORT_LOCAL/health" > /dev/null; then
        echo "  Healthcheck de la API exitoso ✔"
        break
    fi
    echo "  Esperando inicialización de la API (Intento $i/10)..."
    sleep 3
    if [ $i -eq 10 ]; then
        echo "❌ Error: El servicio FastAPI no se desplegó de forma correcta en el puerto $VERIFY_PORT_LOCAL"
        exit 1
    fi
done

echo ""
echo "===================================================================="
echo " DEPLOY EXITOSO — $IMAGE_NAME:$VERSION"
echo " API   : http://localhost:$VERIFY_PORT_LOCAL"
echo " Docs  : http://localhost:$VERIFY_PORT_LOCAL/docs"
echo " MLflow: http://localhost:$VERIFY_PORT_MLFLOW"
echo "===================================================================="
