"""src/manage_versions.py — Gestión de versiones en MLflow Model Registry."""

import sys
import os
import logging

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
import mlflow
from mlflow import MlflowClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s | MANAGE_VERSIONS | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

log = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = getattr(C, "MODEL_NAME", "modelo_renovacion_prestamo")

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)


def listar_versiones(nombre: str) -> None:
    """Lista todas las versiones registradas del modelo en el Registry de MLflow."""
    try:
        versiones = client.search_model_versions(f"name='{nombre}'")
        if not versiones:
            log.warning(f"⚠️ No hay versiones registradas para el modelo '{nombre}'")
            return

        log.info(f"=== Versiones encontradas para '{nombre}' ===")
        for v in versiones:
            log.info(
                f"  📌 v{v.version} | Estado: {v.current_stage:12} | Run ID: {v.run_id[:8]}..."
            )
    except Exception as e:
        log.error(f"❌ Error al listar las versiones del modelo: {e}")


def run() -> None:
    log.info("=== ETAPA 6: MANEJO DE VERSIONES DE MODELOS CON MLFLOW ===")

    if not MODEL_NAME:
        raise ValueError(
            "El nombre del modelo en config.py (MODEL_NAME) no está definido o es inválido."
        )

    log.info(f"Conectando a MLflow Server en: {MLFLOW_URI}")
    log.info(f"Buscando en Model Registry: {MODEL_NAME}")

    # 1. Ver versiones disponibles en el servidor centralizado
    listar_versiones(MODEL_NAME)

    log.info("=== ETAPA 6 COMPLETADA ===")


if __name__ == "__main__":
    run()
