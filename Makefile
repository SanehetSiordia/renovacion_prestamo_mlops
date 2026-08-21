# Makefile — Pipeline CI/CD Local + Desarrollo (Renovacion de Prestamo)

-include .env
export

.PHONY: create-dirs aws-dvc-up download-aws download-dvc dvc-push \
        check-mlflow check-training \
        gcp-service-up validate-gcp-permissions upload-gcp-models register-vertex push-fastapi-gcp all-gcp \
        all train test validate versions \
        dev-up dev-down dev-logs dev-logs-api dev-logs-mlflow dev-ps \
        prod-up prod-down deploy rollback clean-files clean-all help

COMPOSE_FILE		?= compose.yml
COMPOSE_FILE_PROD	?= compose.prod.yml
ENV_PROD			?= .env.prod

LOCAL_DIR_RAW      	?= ./data/raw
LOCAL_DIR_PROCESSED	?= ./data/processed

VERSION				?= $(IMAGE_VERSION)
ifeq ($(VERSION),)
  VERSION			:= latest
endif

MLFLOW_PORT					?= $(MLFLOW_PORT)
IMAGE_TRAIN					?= $(IMAGE_NAME_TRAINING)
IMAGE_SERVING				?= $(IMAGE_NAME_SERVING)
IMAGE_MLFLOW				?= $(IMAGE_NAME_MLFLOW)
IMAGE_DVC					?= $(IMAGE_NAME_DVC)
IMAGE_NAME_GCP_VERTEXAI		?= $(IMAGE_NAME_GCP_VERTEXAI)

DOCKER_TRAINING_NAME 		?= $(DOCKER_TRAINING_NAME)
DOCKER_MLFLOW_NAME 			?= $(DOCKER_MLFLOW_NAME)
DOCKER_GCP_VERTEXAI_NAME	?= $(DOCKER_GCP_VERTEXAI_NAME)

GOOGLE_APPLICATION_CREDENTIALS	?= $(GOOGLE_APPLICATION_CREDENTIALS)
GCP_DESTINATION     			?= $(GCP_DESTINATION)
GCP_CONTAINER_IMAGE_URI 		?= $(GCP_CONTAINER_IMAGE_URI)
GCP_FASTAPI_REMOTE  			?= $(GCP_ARTIFACT_REGISTRY_URI)/$(IMAGE_NAME_FASTAPI):$(IMAGE_VERSION)
GCP_ARTIFACT_REGISTRY_URI		?= $(GCP_ARTIFACT_REGISTRY_URI)

#Nombres de servicios Docker Compose
TRAINING_SERVICE_NAME		?= training_service
GCP_SERVICE_NAME			?= gcp_vertexai_service
MLFLOW_SERVICE_NAME			?= mlflow_service
DVC_AWS_SERVICE_NAME		?= dvc_aws_service
FASTAPI_SERVICE_NAME		?= fastapi_service

# ── 1. Gestion de Datos y Versionado (AWS S3 & DVC) ─────────────────────────────────────────────

create-dirs:
	@echo "=== Verificando directorios necesarios para datos CSV ==="
	mkdir -p $(LOCAL_DIR_RAW)
	mkdir -p $(LOCAL_DIR_PROCESSED)

aws-dvc-up:
	@echo "=== Verificando estado del contenedor dvc_aws ==="
	@if [ -z "$$(docker compose -f $(COMPOSE_FILE) ps -q $(DVC_AWS_SERVICE_NAME) 2>/dev/null)" ]; then \
		echo "=== Levantando contenedor DVC-AWS ==="; \
		docker compose -f $(COMPOSE_FILE) up -d $(DVC_AWS_SERVICE_NAME); \
	fi

download-aws: create-dirs aws-dvc-up
	@echo "=== Descargando datos desde AWS S3 dentro del contenedor ==="
	docker compose -f $(COMPOSE_FILE) exec $(DVC_AWS_SERVICE_NAME) aws s3 cp s3://$(AWS_S3_BUCKET)/$(AWS_RAW_FILE) data/raw/$(AWS_RAW_FILE)
	docker compose -f $(COMPOSE_FILE) exec $(DVC_AWS_SERVICE_NAME) aws s3 cp s3://$(AWS_S3_BUCKET)/$(AWS_PROCESSED_FILE) data/processed/$(AWS_PROCESSED_FILE)

download-dvc: create-dirs aws-dvc-up
	@echo "=== Descargando datos desde S3 utilizando DVC ==="
	docker compose -f $(COMPOSE_FILE) exec $(DVC_AWS_SERVICE_NAME) dvc pull --force

dvc-push: aws-dvc-up
	@echo "=== Subiendo artefactos a S3 mediante DVC ==="
	docker compose -f $(COMPOSE_FILE) exec $(DVC_AWS_SERVICE_NAME) dvc push
	@echo "=== Verificando estado de DVC ==="
	docker compose -f $(COMPOSE_FILE) exec $(DVC_AWS_SERVICE_NAME) dvc status

# ── 2. Pipeline de CI/CD Local (Entrenamiento & Calidad) ──────────────────────────────────────────────────────

check-mlflow:
	@STATUS=$$(docker inspect --format='{{.State.Health.Status}}' $(DOCKER_MLFLOW_NAME) 2>/dev/null || echo "not_found"); \
	if [ "$$STATUS" = "healthy" ]; then \
		echo "Contenedor $(DOCKER_MLFLOW_NAME) listo y saludable."; \
	else \
		echo "Contenedor $(DOCKER_MLFLOW_NAME) no disponible (Estado: $$STATUS). Levantando servicio..."; \
		docker compose -f $(COMPOSE_FILE) up -d --build $(MLFLOW_SERVICE_NAME); \
		echo "=== Esperando que el contenedor $(DOCKER_MLFLOW_NAME) pase el Healthcheck... ==="; \
		until [ "$$(docker inspect --format='{{.State.Health.Status}}' $(DOCKER_MLFLOW_NAME) 2>/dev/null)" = "healthy" ]; do \
			sleep 2; \
		done; \
		echo "Contenedor $(DOCKER_MLFLOW_NAME) listo y saludable para recibir metricas."; \
	fi

check-training: check-mlflow
	@STATUS=$$(docker inspect --format='{{.State.Status}}' $(DOCKER_TRAINING_NAME) 2>/dev/null || echo "not_found"); \
	if [ "$$STATUS" = "running" ]; then \
		echo "Contenedor $(DOCKER_TRAINING_NAME) listo y corriendo."; \
	else \
		echo "Levantando contenedor de entrenamiento..."; \
		docker compose -f $(COMPOSE_FILE) up -d --build $(TRAINING_SERVICE_NAME); \
		echo "=== Esperando que el contenedor $(DOCKER_TRAINING_NAME) se inicie... ==="; \
		until [ "$$(docker inspect --format='{{.State.Status}}' $(DOCKER_TRAINING_NAME) 2>/dev/null)" = "running" ]; do \
			sleep 1; \
		done; \
		echo "Contenedor $(DOCKER_TRAINING_NAME) inicializado."; \
	fi

all: download-dvc check-training train test validate versions
	@echo "====================================================="
	@echo "Pipeline completado. Modelo Entrenado Exportado listo para API"
	@echo "Proceda a ejecutar 'all-gcp' para subir artefactos a GCP y registrar el modelo en Vertex AI"
	@echo "====================================================="

train:
	@echo "=== [Paso 1/4] Procesando datos y entrenando modelo ==="
	docker exec -i $(DOCKER_TRAINING_NAME) python src/manage_data.py
	docker exec -i $(DOCKER_TRAINING_NAME) python src/train_model.py

test:
	@echo "=== [Paso 2/4] Ejecutando pruebas unitarias ==="
	docker exec -i $(DOCKER_TRAINING_NAME) pytest tests/test_data.py -v -s
	docker exec -i $(DOCKER_TRAINING_NAME) pytest tests/test_model.py -v -s
	docker exec -i $(DOCKER_TRAINING_NAME) pytest tests/test_pipeline.py -v -s

validate:
	@echo "=== [Paso 3/4] Quality Gate y Validacion de Metricas ==="
	docker exec -i $(DOCKER_TRAINING_NAME) python src/validate_model.py

versions:
	@echo "=== [Paso 4/4] Registro de version en MLflow ==="
	docker exec -i $(DOCKER_TRAINING_NAME) python src/manage_versions.py

# ── 3. Integracion y Publicacion en Google Cloud Platform (GCP) ──────────────

gcp-service-up:
	@echo "=== Verificando estado del contenedor GCP Vertex AI ==="
	@if [ -z "$$(docker compose -f $(COMPOSE_FILE) ps -q $(GCP_SERVICE_NAME) 2>/dev/null)" ]; then \
		echo "=== Levantando contenedor GCP ==="; \
		docker compose -f $(COMPOSE_FILE) up -d $(GCP_SERVICE_NAME); \
	fi

validate-gcp-permissions: gcp-service-up
	@echo "=== 1. Validando cuenta activa y proyecto en el contenedor ==="
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud config get-value account
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud config set project $(GCP_PROJECT_ID)
	@echo "=== 2. Validando permisos en Google Cloud Storage (Bucket) ==="
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud storage ls $(GCP_DESTINATION)/
	@echo "=== 3. Validando permisos en Artifact Registry ==="
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud artifacts repositories list --location=$(GCP_DEFAULT_REGION)
	@echo "=== 4. Validando permisos en Vertex AI ==="
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud ai models list --region=$(GCP_DEFAULT_REGION)
	@echo " Todos los permisos validados exitosamente en GCP."

upload-gcp-models: validate-gcp-permissions
	@echo "=== Subiendo artefactos a GCS Bucket ==="
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud storage cp artifacts/$(GCP_MODEL_PKL_NAME) $(GCP_DESTINATION)/$(GCP_MODEL_PKL_NAME)
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud storage cp artifacts/$(GCP_MODEL_SKOPS_NAME) $(GCP_DESTINATION)/$(GCP_MODEL_SKOPS_NAME)
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud storage cp artifacts/$(GCP_MODEL_JSON_NAME) $(GCP_DESTINATION)/$(GCP_MODEL_JSON_NAME)
	@echo "=== Listando archivos subidos ==="
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud storage ls $(GCP_DESTINATION)/

register-vertex: validate-gcp-permissions
	@echo "=== Registrando modelo en Vertex AI Model Registry ==="
	MSYS_NO_PATHCONV=1 docker compose -f $(COMPOSE_FILE) exec $(GCP_SERVICE_NAME) \
		gcloud ai models upload \
			--region=$(GCP_DEFAULT_REGION) \
			--display-name="$(MODEL_NAME)" \
			--container-image-uri="$(GCP_CONTAINER_IMAGE_URI)" \
			--artifact-uri="$(GCP_DESTINATION)/" \
			--description="Modelo para prediccion de renovacion de prestamos."

push-fastapi-gcp: validate-gcp-permissions
	@echo "=== 1. Autenticando Docker con Artifact Registry de GCP ==="
	gcloud auth configure-docker $(GCP_DEFAULT_REGION)-docker.pkg.dev --quiet
	@echo "=== 2. Construyendo imagen de FastAPI ==="
	docker compose -f $(COMPOSE_FILE) build $(FASTAPI_SERVICE_NAME)
	@echo "=== 3. Etiquetando imagen para GCP Artifact Registry ==="
	docker tag $(IMAGE_NAME_FASTAPI):$(IMAGE_VERSION) $(GCP_FASTAPI_REMOTE)
	@echo "=== 4. Subiendo imagen ==="
	docker push $(GCP_FASTAPI_REMOTE)

all-gcp: upload-gcp-models register-vertex push-fastapi-gcp
	@echo "====================================================="
	@echo "Artefactos Registrados del Modelo Entrenado en GCP"
	@echo "====================================================="

# ── 6. Limpieza y Mantenimiento ────────────────────────────────────
down:
	docker compose -f $(COMPOSE_FILE) down -v
	docker builder prune -f
	@echo "=== Entorno detenido, volumenes y caches purgados ==="

clean-files:
	@echo "=== Limpiando caches y temporales ==="
	rm -rf artifacts/* data/processed/* mlruns/* .pytest_cache htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean-files
	@echo "=== Purgando imagenes huerfanas y caches de Docker ==="
	docker image prune -f
	docker builder prune -f
	docker compose -f $(COMPOSE_FILE) down -v


# ── 8. Ayuda en Consola ──────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "===================================================================="
	@echo "   Opciones de Automatizacion del Makefile — Pipeline MLOps Integral "
	@echo "===================================================================="
	@echo "1. Gestion de Datos y Versionado (AWS S3 & DVC):"
	@echo "  make create-dirs              — Crea la estructura de carpetas locales (data/raw, data/processed)"
	@echo "  make aws-dvc-up               — Verifica e inicia el contenedor de DVC / AWS CLI"
	@echo "  make download-aws             — Descarga directa de archivos CSV desde AWS S3"
	@echo "  make download-dvc             — Descarga datos versionados mediante 'dvc pull'"
	@echo "  make dvc-push                 — Sube nuevos conjuntos de datos a S3 con 'dvc push'"
	@echo ""
	@echo "2. Pipeline de CI/CD Local (Entrenamiento & Calidad):"
	@echo "  make check-mlflow             — Valida o inicia el servidor de MLflow Tracking con Healthcheck"
	@echo "  make check-training           — Valida o inicia el contenedor de entrenamiento"
	@echo "  make all                      — Orquesta el flujo completo: DVC -> Train -> Tests -> Validate -> Versions"
	@echo "  make train                    — Procesa datos y ejecuta el entrenamiento del modelo XGBoost"
	@echo "  make test                     — Ejecuta pruebas unitarias de datos, modelo y pipeline (Pytest)"
	@echo "  make validate                 — Aplica el Quality Gate de metricas sobre los artefactos"
	@echo "  make versions                 — Registra el nuevo modelo y artefactos en MLflow Model Registry"
	@echo ""
	@echo "3. Integracion y Publicacion en Google Cloud Platform (GCP):"
	@echo "  make gcp-service-up           — Inicia el contenedor gestor de GCP con sesion ADC de Windows"
	@echo "  make validate-gcp-permissions — Valida permisos en GCS Bucket, Artifact Registry y Vertex AI"
	@echo "  make upload-gcp-models        — Sube artefactos (.pkl, .skops, .json) al bucket de GCS"
	@echo "  make register-vertex          — Registra el modelo en Vertex AI Model Registry con contenedor oficial"
	@echo "  make push-fastapi-gcp         — Compila, etiqueta y sube la imagen Docker de FastAPI a Artifact Registry"
	@echo "  make all-gcp                  — Exportacion integral a GCP: GCS + Vertex AI + Artifact Registry"
	@echo ""
	@echo "4. Limpieza y Mantenimiento:"
	@echo "  make down                     — Detiene el entorno y purga volumenes de Docker Compose"
	@echo "  make clean-files              — Elimina artefactos, temporales, caches de Python y cobertura"
	@echo "  make clean-all                — Limpieza total: archivos + prune de imagenes y builder cache de Docker"
	@echo "===================================================================="