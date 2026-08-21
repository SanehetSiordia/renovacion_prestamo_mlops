"""
validate_model.py — Valida que las métricas superen el umbral mínimo.

Si falla, el pipeline CI/CD se detiene y no se construye la imagen Docker.
"""
import sys
import logging

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s | VALIDATE_MODEL | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

log = logging.getLogger(__name__)

# ── Funciones para gestionar metricas json ──────────────────────────────────────


def obtener_metricas_json(metrics_path: Path = C.METRICS_PATH) -> tuple[dict, dict, dict]:
    metricas_dict = dict()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metricas_dict = json.load(f)

    log.info(f'📊 ARCHIVO DE MÉTRICAS DETECTADO - RESUMEN DE RESULTADOS:\n{json.dumps(metricas_dict, indent=4, ensure_ascii=False)}')
    algoritmo_modelo = {"algoritmo": metricas_dict["algoritmo"]}
    hiperparametros_modelo = metricas_dict["hiperparametros_finales"]
    resultados_modelo = metricas_dict["metricas_evaluacion"]

    log.info(f"🔹 Modelo: {algoritmo_modelo}")
    log.info(f"🔹 Hiperparámetros: {hiperparametros_modelo}")
    log.info(f"🔹 Métricas: {resultados_modelo}")

    return algoritmo_modelo, hiperparametros_modelo, resultados_modelo


def quality_gate(algoritmo: dict, hiperparametros: dict, metricas: dict) -> None:

    umbral = C.UMBRAL_MIN
    algoritmo = algoritmo["algoritmo"]
    accuracy = metricas["accuracy"]
    f1_score = metricas["f1_score"]
    recall = metricas["recall"]
    roc_auc = metricas["roc_auc"]

    print("=" * 50)
    print(" QUALITY GATE — VALIDACIÓN DE MÉTRICAS")
    print("=" * 50)
    print(f" Algormitmo del Modelo   : {algoritmo}")
    print(f" Recall   : {recall:.4f}  (umbral: >= {umbral})")
    print(f" F1-Score : {f1_score:.4f}")
    print(f" Accuracy : {accuracy:.4f}")
    print(f" Roc_AUC : {roc_auc:.4f}")

    if recall < umbral:
        print(f"\n FALLO: Recall {recall:.4f} < umbral {umbral}")
        print(" El pipeline CI/CD se detiene..")
        sys.exit(1)

    print("=" * 50)
    print(f"\n APROBADO: Recall {recall:.4f} >= {umbral}")
    print(" Pipeline CI/CD continúa.")
    print("=" * 50)

# ── Función principal ─────────────────────────────────────────────────────


def run():
    log.info('=== ETAPA 5: VALIDACION DE RESULTADOS DEL MODELO ===')

    if not C.METRICS_PATH.exists():
        raise FileNotFoundError(f'Ruta del metricas no encontradas: {C.METRICS_PATH}')
        sys.exit(1)

    algoritmo, hiperparametros, metricas = obtener_metricas_json()
    quality_gate(algoritmo, hiperparametros, metricas)

    log.info('=== ETAPA 5 COMPLETADA ===')
    return None


if __name__ == '__main__':
    run()
