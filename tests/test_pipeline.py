"""tests/test_model.py — Tests para revisar el modelo entrenado."""
import sys
import logging

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
import pytest
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s | TEST_PIPELINE | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

log = logging.getLogger(__name__)

# ── FIXTURES DE CONFIGURACIÓN ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def metricas_modelo() -> tuple[dict, dict]:
    """Cargado las metricas obtenidas del modelo."""
    if not C.METRICS_PATH.exists():
        raise FileNotFoundError(f'Metricas no encontradas en la ruta: {C.METRICS_PATH}')

    with open(C.METRICS_PATH, "r", encoding="utf-8") as f:
        metricas_dict = json.load(f)

    return metricas_dict

# ── MÉTODOS DE PRUEBA (TESTS) ────────────────────────────────────────────────


def test_train_returns_metrics(metricas_modelo):
    """El modelo debe retornar las métricas esperadas."""
    metricas = metricas_modelo["metricas_evaluacion"]
    assert "f1_score" in metricas
    assert "recall" in metricas
    assert "accuracy" in metricas
    assert "roc_auc" in metricas


def test_train_f1_positive(metricas_modelo):
    """Las metricas deben ser mayor que 0."""
    metricas = metricas_modelo["metricas_evaluacion"]
    assert metricas["f1_score"] > 0.0
    assert metricas["accuracy"] > 0.0
    assert metricas["recall"] > 0.0
    assert metricas["roc_auc"] > 0.0


def test_train_metrics_in_range(metricas_modelo):
    """Las métricas deben estar en el rango [0, 1]."""
    metricas = metricas_modelo["metricas_evaluacion"]
    for key in ("f1_score", "recall", "accuracy", "roc_auc"):
        assert 0.0 <= metricas[key] <= 1.0, f"{key} fuera de rango: {metricas[key]}"


def test_train_saves_best_params(metricas_modelo):
    """metrics.json debe incluir los mejores parámetros del GridSearch."""
    metricas = metricas_modelo
    assert "hiperparametros_finales" in metricas
    assert isinstance(metricas["hiperparametros_finales"], dict)
