"""tests/test_model.py — Tests para revisar el modelo entrenado."""
import sys
import logging

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

import pytest
import pickle
import numpy as np
import pandas as pd
from typing import Dict
from typing import Any
from sklearn.base import ClassifierMixin

logging.basicConfig(level=logging.INFO, format='%(asctime)s | TEST_MODEL | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

log = logging.getLogger(__name__)

# ── MODELO DE PRUEBA (MOCK) ──────────────────────────────────────────────────


class DummyModel:
    """Clase Mock para pruebas unitarias rápidas de la firma del modelo."""

    def predict_proba(self, X: pd.DataFrame):
        return [[0.35, 0.65]]

# ── FIXTURES DE CONFIGURACIÓN ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def modelo_entrenado() -> ClassifierMixin:
    """Cargando el modelo pickle para los metodos de pruebas."""
    if not C.MODEL_PKL_PATH.exists():
        raise FileNotFoundError(f'Modelo no encontrado en la ruta: {C.MODEL_PKL_PATH}')

    with open(C.MODEL_PKL_PATH, 'rb') as archivo:
        modelo_cargado = pickle.load(archivo)

    log.info(f'[Fixture] Modelo cargado exitosamente desde: {C.MODEL_PKL_PATH.name}')
    return modelo_cargado


@pytest.fixture(scope="module")
def payload_dummy() -> Dict[str, Any]:
    """Genera dinámicamente un diccionario dummy inicializado en 0 basado en C.FEATURES. """
    datos_dummy = {str(feature): 0 for feature in C.FEATURES}
    log.info(f'[Fixture] Payload dummy generado con {len(datos_dummy)}')
    return datos_dummy


@pytest.fixture(scope="module")
def X_test_dummy(payload_dummy: dict) -> pd.DataFrame:
    """Transforma el payload sintético en el DataFrame exacto que espera el modelo."""
    datos_alineados = {f: payload_dummy.get(f, 0) for f in C.FEATURES}
    return pd.DataFrame([datos_alineados])[C.FEATURES]

# ── MÉTODOS DE PRUEBA (TESTS) ────────────────────────────────────────────────


def test_model_has_predict(modelo_entrenado):
    """El modelo debe tener el método predict."""
    assert hasattr(modelo_entrenado, "predict")


def test_model_has_predict_proba(modelo_entrenado):
    """El modelo debe tener el método predict_proba."""
    assert hasattr(modelo_entrenado, "predict_proba")


def test_model_predicts_binary(modelo_entrenado, X_test_dummy):
    y_pred = modelo_entrenado.predict(X_test_dummy)
    assert set(y_pred).issubset({0, 1})


def test_model_predict_proba_shape(modelo_entrenado, X_test_dummy):
    """predict_proba debe retornar shape (n, 2) con valores en [0, 1]."""
    proba = modelo_entrenado.predict_proba(X_test_dummy)
    assert proba.shape == (1, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_predict_proba_sums_to_one(modelo_entrenado, X_test_dummy):
    """Las probabilidades por fila deben sumar 1."""
    proba = modelo_entrenado.predict_proba(X_test_dummy)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_model_is_valid_tree_algorithm(modelo_entrenado):
    """El modelo debe ser un modelo supervisado de decision."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    assert isinstance(
        modelo_entrenado,
        (DecisionTreeClassifier, RandomForestClassifier, XGBClassifier)
    ), f"El modelo entrenado es de un tipo no permitido: {type(modelo_entrenado).__name__}"
