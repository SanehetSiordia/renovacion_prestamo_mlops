"""
manage_data.py —  Ingestión, tratamiento de datos y Feature engineering.
Entrada : data/raw/raw_renovacion_prestamo.csv
Salida  : data/processed/processed_renovacion_prestamo.csv
"""

import sys
import logging

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s | MANAGE_DATA | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

log = logging.getLogger(__name__)

# ── Funciones de limpieza ─────────────────────────────────────────────────


def renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, col_nueva in C.COLUMNAS_RENOMBRAR.items():
        if col in df.columns:
            df = df.rename(columns={col: col_nueva})
            log.info(f'Columna renombrada: {col} -> {col_nueva}')
    return df


def imputar_columnas_negativas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in C.COLUMNAS_IMPUTAR_NEGATIVOS:
        df[col] = np.maximum(0, df[col])
        log.info(f'Columna con negativos imputados a 0: {col}')
    return df


def transformar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in C.COLUMNAS_TRANSFORMAR_LOG:
        new_col_name = f'{col}_LOG'
        if col in df.columns:
            df[new_col_name] = np.log1p(df[col])
            log.info(f'Columna transformada con log1p: {col} -> {new_col_name}')
    return df


def imputar_nulos_sampling_aleatorio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in C.COLUMNAS_NULOS_SAMPLING_ALEATORIO:
        if col in df.columns:
            null_indices = df[col].isnull()
            random_imputed_values = np.random.uniform(max(0, (df[col].mean() - df[col].std())),
                                                      (df[col].mean() + df[col].std()),
                                                      null_indices.sum())
            df.loc[null_indices, col] = random_imputed_values
            log.info(f'Columna con nulos imputados con Sampling Aleatorio: {col}')
            log.info(f'Nulos restantes en {col}: {df[col].isna().sum()}')
    return df


def imputar_nulos_mediana(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in C.COLUMNAS_NULOS_MEDIANA:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            log.info(f'Columna con nulos imputados con la Mediana: {col}')
            log.info(f'Nulos restantes en {col}: {df[col].isna().sum()}')
    return df


def imputar_nulos_categorias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in C.COLUMNAS_CATEGORIAS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            log.info(f'Columna de variables categoricas con nulos imputados: {col}')
            log.info(f'Nulos restantes en {col}: {df[col].isna().sum()}')
    return df

# ── Funciones de preparación ──────────────────────────────────────────────


def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=C.COLUMNAS_CATEGORIAS, drop_first=False, dtype=int)
    log.info(f'Total de columnas antes del One-Hot-Encoding: {df.shape[1]}')
    log.info(f'Total de columnas despues del One-Hot-Encoding: {df_encoded.shape[1]}')
    return df_encoded


def clustering_kmeans(df_encoded: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df_encoded.copy()
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(df_encoded[C.COLUMNAS_KMEANS].copy())
    df_encoded['Cluster'] = kmeans.labels_
    log.info(f'Total de filas y columnas con Cluster Kmeans: {df_encoded.shape}')
    return df_encoded


def procesed_data(df_encoded: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df_encoded.copy()
    existing_cols_to_drop = [col for col in C.COLUMNAS_DROP if col in df_encoded.columns]
    df_processed = df_encoded.drop(columns=existing_cols_to_drop)
    log.info(f'Total de filas y columnas despues del Procesamiento de variables: {df_processed.shape}')
    return df_processed


# ── Función principal ─────────────────────────────────────────────────────

def run():
    log.info('=== ETAPA 1: Ingestión y tratamientos de datos ===')
    if not C.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset no encontrado: {C.RAW_DATA_PATH}')
    df = pd.read_csv(C.RAW_DATA_PATH, sep=';')
    log.info(f'Cargado Dataframe: {df.shape[0]} filas x {df.shape[1]} columnas')

    df = renombrar_columnas(df)
    df = imputar_columnas_negativas(df)
    df = transformar_columnas(df)
    df = imputar_nulos_sampling_aleatorio(df)
    df = imputar_nulos_mediana(df)
    df = imputar_nulos_categorias(df)

    log.info('=== ETAPA 1 COMPLETADA ===')
    log.info('=== ETAPA 2: Feature engineering y selección de variables ===')

    df_encoded = one_hot_encoding(df)
    df_encoded = clustering_kmeans(df_encoded)
    df_processed = procesed_data(df_encoded)

    C.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(C.PROCESSED_DATA_PATH, index=False)

    log.info(f'Artefacto guardado: {C.PROCESSED_DATA_PATH}')
    log.info('=== ETAPA 2 COMPLETADA ===')
    return None


if __name__ == '__main__':
    run()
