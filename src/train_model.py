"""
train_model.py — entrena el modelo y crea los artefactos correspondientes.
Entrada : data/processed/processed_renovacion_prestamo.csv
Salida  : artifacts/metrics.json | artifacts/model.pkl | artifacts/model.skops
"""
import sys
import os
import logging
import warnings

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from typing import Any
from typing import cast
from typing import Tuple
from mlflow.models import infer_signature
import mlflow.sklearn as ml_learn
import mlflow
import socket
import json
import pickle
import skops.io as sio
import numpy as np
import pandas as pd

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | TRAIN_DATA | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

log = logging.getLogger(__name__)


# ── Declaracion de Constantes para MODELAR ───────────────────────────────────────
MODELOS = {
    'DecisionTree': DecisionTreeClassifier(random_state=C.RANDOM_STATE),
    'RandomForest': RandomForestClassifier(random_state=C.RANDOM_STATE, n_estimators=10, max_depth=4),
    'XGBoost': XGBClassifier(random_state=C.RANDOM_STATE, n_estimators=10, max_depth=4, learning_rate=0.1, subsample=0.5)}

HIPERPARAMETROS = {
    "DecisionTree": {
        "random_state": [C.RANDOM_STATE],
        "max_depth": [4, 6, 8, 10, None],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 10],
    },
    "RandomForest": {
        "random_state": [C.RANDOM_STATE],
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"],
    },
    "XGBoost": {
        "random_state": [C.RANDOM_STATE],
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.5, 0.8, 1.0],
    },
}

# ── Funciones de balanceo de datos ───────────────────────────────────────


def separar_datos_entrenamiento(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    X = df.drop(columns=[C.TARGET])
    y = df[C.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y)
    df_train = pd.concat([X_train, y_train], axis=1)

    log.info(f'Datos divididos con éxito (test_size={C.TEST_SIZE})')
    log.info(f'X_train: {X_train.shape} | X_test: {X_test.shape}')
    log.info(f'y_train: {y_train.shape} | y_test: {y_test.shape}')
    log.info(f'Distribucion Target en y_train: {y_train.value_counts(normalize=True)}')
    log.info(f'Distribucion Target en y_test: {y_test.value_counts(normalize=True)}')
    log.info(f'Distribucion Target en df_train: {df_train[C.TARGET].value_counts()}')
    return df_train, X_train, X_test, y_train, y_test


def balancear_datos_undersampling(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    sampler = RandomUnderSampler(sampling_strategy='auto', random_state=C.RANDOM_STATE)
    X_train_under, y_train_under = cast(Tuple[Any, Any], sampler.fit_resample(X_train, y_train))
    df_train_under = pd.concat([X_train_under, y_train_under], axis=1)
    log.info('Datos Balanceados con éxito (UNDERSAMPLING)')
    log.info(f'X_train_under: {X_train_under.shape}')
    log.info(f'y_train_under: {y_train_under.shape}')
    log.info(f'Distribucion Target en df_train_under: {df_train_under[C.TARGET].value_counts()}')
    return df_train_under


def balancear_datos_oversampling(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    sampler = RandomOverSampler(sampling_strategy='auto', random_state=C.RANDOM_STATE)
    X_train_over, y_train_over = cast(Tuple[Any, Any], sampler.fit_resample(X_train, y_train))
    df_train_over = pd.concat([X_train_over, y_train_over], axis=1)
    log.info('Datos Balanceados con éxito (OVERSAMPLING)')
    log.info(f'X_train_over: {X_train_over.shape}')
    log.info(f'y_train_over: {y_train_over.shape}')
    log.info(f'Distribucion Target en df_train_over: {df_train_over[C.TARGET].value_counts()}')
    return df_train_over


def balancear_datos_smote(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    sampler = SMOTE(sampling_strategy='auto', random_state=C.RANDOM_STATE)
    X_train_smote, y_train_smote = cast(Tuple[Any, Any], sampler.fit_resample(X_train, y_train))
    df_train_smote = pd.concat([X_train_smote, y_train_smote], axis=1)
    log.info('Datos Balanceados con éxito (SMOTE)')
    log.info(f'X_train_smote: {X_train_smote.shape}')
    log.info(f'y_train_smote: {y_train_smote.shape}')
    log.info(f'Distribucion Target en df_train_smote: {df_train_smote[C.TARGET].value_counts()}')
    return df_train_smote


# ── Funciones de Modelado con MLFLOW ──────────────────────────────────────
def modelar_mlflow(df_train: pd.DataFrame,
                   df_train_under: pd.DataFrame,
                   df_train_over: pd.DataFrame,
                   df_train_smote: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   ) -> Tuple[ClassifierMixin, str, pd.DataFrame, str, float]:
    datasets_entrenamiento = {
        "original": (
            df_train.drop(columns=[C.TARGET]),
            df_train[C.TARGET],
        ),
        "undersampling": (
            df_train_under.drop(columns=[C.TARGET]),
            df_train_under[C.TARGET],
        ),
        "oversampling": (
            df_train_over.drop(columns=[C.TARGET]),
            df_train_over[C.TARGET],
        ),
        "smote": (
            df_train_smote.drop(columns=[C.TARGET]),
            df_train_smote[C.TARGET],
        ),
    }

    uri_tracking = C.MLFLOW_URI

    if "http://mlflow:" in uri_tracking:
        try:
            socket.getaddrinfo("mlflow", 5000)
            log.info("📡 Entorno Container detectado. Usando URI nativa: mlflow")
        except socket.gaierror:
            uri_tracking = uri_tracking.replace("http://mlflow:", "http://localhost:")
            log.info(f"⚠️ Entorno Host/Codespace detectado. Conmutando URI global a: {uri_tracking}")

    os.environ["MLFLOW_TRACKING_URI"] = uri_tracking
    mlflow.set_tracking_uri(uri_tracking)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=C.MLFLOW_RUN_NAME):
        mlflow.log_params(
            {
                "k_features": C.K_FEATURES,
                "test_size": C.TEST_SIZE,
                "random_state": C.RANDOM_STATE,
            }
        )

        mejor_recall = -1.0
        mejor_modelo_global = None
        mejor_nombre_modelo = ""
        mejor_tipo_dataset = ""
        metricas_ganadoras = {}
        df_train_best = pd.DataFrame()

        for nombre_modelo, instancia_modelo in MODELOS.items():
            log.info(f"--- Evaluando algoritmo: {nombre_modelo} ---")

            for tipo_dataset, (X_train_v, y_train_v) in datasets_entrenamiento.items():
                modelo_clonado = clone(instancia_modelo)
                modelo_clonado.fit(X_train_v, y_train_v)

                y_pred = modelo_clonado.predict(X_test)
                accuracy = np.round(accuracy_score(y_test, y_pred), 4)
                f1 = np.round(f1_score(y_test, y_pred, zero_division=0), 4)
                recall = np.round(recall_score(y_test, y_pred, zero_division=0), 4)
                roc_auc = np.round(roc_auc_score(y_test, y_pred), 4)

                log.info(
                    f"Dataset {tipo_dataset} -> F1={f1} | Recall={recall} | Accuracy={accuracy} | Roc AUC={roc_auc}"
                )

                if recall > mejor_recall:
                    mejor_recall = recall
                    mejor_modelo_global = modelo_clonado
                    mejor_nombre_modelo = nombre_modelo
                    mejor_tipo_dataset = tipo_dataset
                    metricas_ganadoras = {
                        "accuracy": accuracy,
                        "f1": f1,
                        "recall": recall,
                        "roc_auc": roc_auc,
                    }
                    df_train_best = pd.concat([X_train_v, y_train_v], axis=1)

        log.info(
            f"=== GANADOR ABSOLUTO: {mejor_nombre_modelo} usando dataset {mejor_tipo_dataset} con RECALL={mejor_recall} ==="
        )

        mlflow.log_metrics(metricas_ganadoras)
        mlflow.log_params(
            {
                "mejor_algoritmo": mejor_nombre_modelo,
                "mejor_dataset_balanceo": mejor_tipo_dataset,
            }
        )
        ml_learn.log_model(
            mejor_modelo_global,
            name="mejor_modelo",
            serialization_format="skops",
            skops_trusted_types=[
                "xgboost.sklearn.XGBClassifier",
                "xgboost.core.Booster",
            ]
        )

    return cast(ClassifierMixin, mejor_modelo_global), mejor_nombre_modelo, df_train_best, mejor_tipo_dataset, mejor_recall


def optimizar_hiperparametros_mlflow(
    mejor_modelo_base: ClassifierMixin,
    nombre_modelo: str,
    mejor_dataset: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    nombre_mejor_balanceo: str,
    recall_base: float,
) -> Tuple[ClassifierMixin, dict]:

    log.info(
        f"=== INICIANDO AJUSTE FINO (TUNING) PARA: {nombre_modelo} ==="
    )

    metricas_modelo = dict()

    X_train_opt = mejor_dataset.drop(columns=[C.TARGET])
    y_train_opt = mejor_dataset[C.TARGET]

    param_grid = HIPERPARAMETROS.get(nombre_modelo, {})

    if not param_grid:
        log.warning(
            f"No se encontraron hiperparámetros para {nombre_modelo}. Se retornará el modelo base."
        )
        return mejor_modelo_base, metricas_modelo

    uri_tracking = C.MLFLOW_URI

    if "http://mlflow:" in uri_tracking:
        try:
            socket.getaddrinfo("mlflow", 5000)
            log.info("📡 Entorno Container detectado. Usando URI nativa: mlflow")
        except socket.gaierror:
            uri_tracking = uri_tracking.replace("http://mlflow:", "http://localhost:")
            log.info(f"⚠️ Entorno Host/Codespace detectado. Conmutando URI global a: {uri_tracking}")

    os.environ["MLFLOW_TRACKING_URI"] = uri_tracking
    mlflow.set_tracking_uri(uri_tracking)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    run_name_tuning = f"Tuning_{nombre_modelo}_{C.MLFLOW_RUN_NAME}"

    with mlflow.start_run(run_name=run_name_tuning):
        grid_search = GridSearchCV(
            estimator=cast(BaseEstimator, mejor_modelo_base),
            param_grid=param_grid,
            scoring="recall",
            cv=C.K_FEATURES,
            n_jobs=1,
            verbose=1,
        )

        log.info(
            f"Buscando la mejor combinación en {param_grid} mediante Cross-Validation..."
        )
        grid_search.fit(X_train_opt, y_train_opt)

        modelo_tuning = cast(ClassifierMixin, grid_search.best_estimator_)
        mejores_params = grid_search.best_params_

        log.info(f"Tuning Completado - Mejores parámetros: {mejores_params}")

        y_pred = modelo_tuning.predict(X_test)

        accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
        f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
        recall = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
        roc_auc = np.round(roc_auc_score(y_test, y_pred), 4)

        log.info(
            f"Resultado Optimizado -> F1={f1} | RECALL OPTIMIZADO={recall} | Accuracy={accuracy} | Roc AUC={roc_auc}"
        )

        # Comparativa de modelo base vs modelo tunning
        if recall > recall_base:
            log.info(f"El Tuning mejoró el Recall base ({recall} > {recall_base}). Seleccionando modelo optimizado.")
            modelo_final = modelo_tuning
            hiperparametros_finales = mejores_params
            es_optimizado = True
        else:
            log.warning(
                f"El Tuning sufrió sobreajuste ({recall} <= {recall_base}). "
                f"Se descarta el tuning y se selecciona el MODELO BASE ORIGINAL."
            )
            modelo_final = mejor_modelo_base
            hiperparametros_finales = mejor_modelo_base.get_params() if hasattr(mejor_modelo_base, 'get_params') else {}

            y_pred_base = mejor_modelo_base.predict(X_test)
            accuracy = round(float(accuracy_score(y_test, y_pred_base)), 4)
            f1 = round(float(f1_score(y_test, y_pred_base, zero_division=0)), 4)
            recall = recall_base
            roc_auc = round(float(roc_auc_score(y_test, y_pred_base)), 4)
            es_optimizado = False

        mlflow.log_params(hiperparametros_finales)
        mlflow.log_param("algoritmo_optimizado", nombre_modelo)
        mlflow.log_param("tuning_aplicado", str(es_optimizado))
        mlflow.log_metrics({"accuracy": accuracy, "f1": f1, "recall": recall, "roc_auc": roc_auc})

        # ── registro de firmas y metadatos en MLFLOW ───────────────────
        y_pred_final = modelo_final.predict(X_test)
        y_pred_df = pd.DataFrame(y_pred_final, columns=[C.TARGET], index=X_test.index)
        firma_modelo = infer_signature(X_test, y_pred_df)

        nombre_artefacto = "modelo_optimizado" if es_optimizado else "modelo_base_ganador"
        model_info = ml_learn.log_model(
            modelo_final,
            name=nombre_artefacto,
            serialization_format="skops",
            skops_trusted_types=[
                "xgboost.sklearn.XGBClassifier",
                "xgboost.core.Booster",
            ],
            signature=firma_modelo,
        )

        log.info("Registrando versión del modelo en el Registry de forma explícita...")
        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=C.MODEL_NAME
        )

        ultima_version = model_version.version
        log.info(f"Creada de forma exitosa la versión '{ultima_version}' para {C.MODEL_NAME}")

        client = mlflow.MlflowClient()

        origen_txt = "Optimizado mediante GridSearchCV" if es_optimizado else "Base (Parámetros por defecto/exploración)"
        texto_descripcion = (
            f"Modelo {nombre_modelo}. Tipo: {origen_txt}.\n"
            f"Estrategia de balanceo de datos de entrenamiento: {nombre_mejor_balanceo}.\n"
            f"Métricas finales en Test -> RECALL: {recall} | F1: {f1}.\n"
        )

        client.update_model_version(
            name=C.MODEL_NAME,
            version=ultima_version,
            description=texto_descripcion
        )

        client.set_model_version_tag(name=C.MODEL_NAME, version=ultima_version, key="autor", value="Sinhue")
        client.set_model_version_tag(name=C.MODEL_NAME, version=ultima_version, key="tipo_balanceo", value=str(nombre_mejor_balanceo))
        client.set_model_version_tag(name=C.MODEL_NAME, version=ultima_version, key="recall_test", value=str(recall))
        client.set_model_version_tag(name=C.MODEL_NAME, version=ultima_version, key="tuning_exitoso", value=str(es_optimizado))

        # Diccionario de salida estructurado
        metricas_modelo = {
            "algoritmo": nombre_modelo,
            "tuning_aplicado": es_optimizado,
            "hiperparametros_finales": hiperparametros_finales,
            "metricas_evaluacion": {
                "accuracy": accuracy,
                "f1_score": f1,
                "recall": recall,
                "roc_auc": roc_auc,
            },
        }
    return modelo_final, metricas_modelo

# ── Funciones para Publicacion ────────────────────────────────────────────


def exportar_modelo_resultados(modelo: ClassifierMixin, datos_json: dict) -> None:

    log.info(f"=== EXPORTANDO ARTEFACTOS LOCALES A {C.ARTIFACTS_DIR} ===")

    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(C.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(datos_json, f, ensure_ascii=False, indent=4)
    log.info(f"Metadata guardada exitosamente en: {C.METRICS_PATH}")

    with open(C.MODEL_PKL_PATH, "wb") as f:
        pickle.dump(modelo, f)
    log.info(f"Modelo guardado en formato Pickle en: {C.MODEL_PKL_PATH}")
    
    sio.dump(modelo, C.MODEL_SKOPS_PATH)
    log.info(f"Modelo guardado en formato Skops en: {C.MODEL_SKOPS_PATH}")

    if isinstance(modelo, XGBClassifier):
        modelo.get_booster().save_model(C.MODEL_JSON_PATH)
        log.info(f"Modelo XGBoost guardado nativamente en JSON en: {C.MODEL_JSON_PATH}")


# ── Función principal ─────────────────────────────────────────────────────

def run():
    log.info('=== ETAPA 3: ENTRENAMIENTO Y SELECCION DEL MEJOR MODELO ===')
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    if not C.PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset no encontrado: {C.PROCESSED_DATA_PATH}')
    df = pd.read_csv(C.PROCESSED_DATA_PATH, sep=',')
    log.info(f'Cargado Dataframe: {df.shape[0]} filas x {df.shape[1]} columnas')

    # 1. Separación de datos y balanceos correspondientes
    df_train, X_train, X_test, y_train, y_test = separar_datos_entrenamiento(df)
    df_train_under = balancear_datos_undersampling(X_train, y_train)
    df_train_over = balancear_datos_oversampling(X_train, y_train)
    df_train_smote = balancear_datos_smote(X_train, y_train)

    # 2. Selección de mejor algoritmo de ML en MLflow
    mejor_modelo, nombre_modelo, mejor_dataset, nombre_mejor_balanceo, mejor_recall = modelar_mlflow(df_train, df_train_under, df_train_over, df_train_smote, X_test, y_test)

    # 3. Ajuste fino del mejor modelo mediante GridSearchCV en MLflow
    modelo_tunning, metricas_modelo = optimizar_hiperparametros_mlflow(mejor_modelo, nombre_modelo, mejor_dataset, X_test, y_test, nombre_mejor_balanceo, mejor_recall)

    log.info('=== ETAPA 3 COMPLETADA ===')
    log.info('=== ETAPA 4: EXPORTACION DEL MEJOR MODELO Y SUS RESULTADOS ===')

    # 4. Exportación física de artefactos (.json, .pkl, .skops)
    exportar_modelo_resultados(modelo_tunning, metricas_modelo)
    log.info('=== ETAPA 4 COMPLETADA ===')
    return None


if __name__ == '__main__':
    run()
