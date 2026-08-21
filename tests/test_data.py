"""tests/test_data.py — Tests para revisar el dataset procesad."""

import pandas as pd
import sys
import logging

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
import pytest

logging.basicConfig(level=logging.INFO, format='%(asctime)s | TEST_DATA | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

log = logging.getLogger(__name__)

# ── FIXTURES DE CONFIGURACIÓN ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dataset_procesado() -> pd.DataFrame:
    """Cargado del dataset procesado una sola vez para todo el módulo de pruebas."""
    if not C.PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset no encontrado en la ruta: {C.PROCESSED_DATA_PATH}')

    df = pd.read_csv(C.PROCESSED_DATA_PATH, sep=',')
    log.info(f'[Fixture] Cargado Dataframe en caché: {df.shape[0]} filas x {df.shape[1]} columnas')
    return df


# ── MÉTODOS DE PRUEBA (TESTS) ────────────────────────────────────────────────
def test_generate_has_correct_columns(dataset_procesado: pd.DataFrame):
    """El dataset debe tener todas las columnas esperadas."""
    df = dataset_procesado
    for col in C.FEATURES + [C.TARGET]:
        assert col in df.columns, f"Columna faltante: {col}"


def test_generate_target_is_binary(dataset_procesado: pd.DataFrame):
    """El Target debe ser binario (0 o 1)."""
    df = dataset_procesado
    assert set(df[C.TARGET].unique()).issubset({0, 1})


def test_generate_class_imbalance(dataset_procesado: pd.DataFrame):
    """La tasa de clase 1 debe estar entre 5% y 20%."""
    df = dataset_procesado
    rate = df[C.TARGET].mean()
    assert 0.025 <= rate <= 0.20, f"Tasa de clase 1 fuera de rango: {rate:.2%}"


def test_generate_no_nulls(dataset_procesado: pd.DataFrame):
    """El dataset no debe tener valores nulos."""
    df = dataset_procesado
    assert df.isnull().sum().sum() == 0
