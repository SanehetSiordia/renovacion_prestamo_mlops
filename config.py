"""
config.py — Configuración centralizada del pipeline MLOps Renovacion de Prestamo
Uso:
    import config as C
"""
import os
from pathlib import Path

# ── Directorios ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_RAW_DIR = ROOT_DIR / 'data/raw'
DATA_PROCESSED_DIR = ROOT_DIR / 'data/processed'
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'
REPORTS_DIR = ROOT_DIR / 'reports'
EVIDENTLY_WORKSPACE_DIR = ROOT_DIR / 'evidently_workspace'

# ── Crear carpetas necesarias ─────────────────────────────────────────────────────────────────
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENTLY_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# ── Archivos ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH = DATA_RAW_DIR      / 'raw_renovacion_prestamo.csv'
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / 'processed_renovacion_prestamo.csv'
REPORT_DRIFT_PATH = REPORTS_DIR / 'reporte_drift_renovacion.html'
JSON_DRIFT_PATH = REPORTS_DIR / 'reporte_drift_renovacion.json'

MODEL_PKL_PATH = ARTIFACTS_DIR / 'model.pkl'
MODEL_SKOPS_PATH = ARTIFACTS_DIR / 'model.skops'
MODEL_JSON_PATH = ARTIFACTS_DIR / 'model.json'
METRICS_PATH = ARTIFACTS_DIR / 'metrics.json'

# ── Columnas  para Preprocesamiento ────────────────────────────────────────────────
TARGET     = 'FLAG_VENTA'
COLUMNAS_RENOMBRAR = {
    "LINEA_RENOVADO": "Linea_Renovado",
    "PLAZO_RENOVADO": "Plazo_Renovado",
    "USO_LINEA_TOTAL_TC_T2": "Uso_Linea",
    "USO_TRIM_LINEA_BBVA": "Uso_TrimLinea",
    "NR_ENTIDADES_TOTAL_T2": "Nro_Entidades",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12": "Dif_Entidades",
    "SDO_CONSUMO_T2": "Saldo_Consumo",
    "RESENCIA_OFERTA_PLD_RENOVADO": "Meses_oferta",
    "Ahorro_Sldo_Bco_T1": "Ahorro",
    "PConsumo_Sldo_Bco_T1": "Prestamo_vigente",
    "SDO_BCO_tot_sm_pasivo_Bco_6M": "Promed_6Mdeuda",
    "FLAG_LIMA_PROVINCIA": "Flag_LimProv",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": "Deuda_Cubierta",
}

COLUMNAS_IMPUTAR_NEGATIVOS = ['Ahorro','Prestamo_vigente','Promed_6Mdeuda']

COLUMNAS_TRANSFORMAR_LOG = ['Uso_Linea',
                            'Uso_TrimLinea',
                            'Saldo_Consumo',
                            'SUELDO_ESTIMADO',
                            'ANTIGUEDAD_MES',
                            'Linea_Renovado',
                            'Ahorro',
                            'Prestamo_vigente',
                            'Promed_6Mdeuda',
                            'SUELDO_ESTIMADO',
                            'Deuda_Cubierta']

COLUMNAS_NULOS_SAMPLING_ALEATORIO=['Uso_TrimLinea_LOG','Uso_Linea_LOG','Meses_oferta']

COLUMNAS_NULOS_MEDIANA=['Saldo_Consumo_LOG',
                        'SUELDO_ESTIMADO_LOG',
                        'ANTIGUEDAD_MES_LOG',
                        'EDAD',
                        'Uso_TrimLinea_LOG',
                        'Prestamo_vigente_LOG',
                        'Uso_Linea_LOG']

COLUMNAS_CATEGORIAS = ['REGION','SEXO','EST_CIVIL']

COLUMNAS_KMEANS = ['Uso_TrimLinea_LOG', 'Prestamo_vigente_LOG', 'Uso_Linea_LOG']

COLUMNAS_DROP=['Uso_Linea',
               'Uso_TrimLinea',
               'Saldo_Consumo',
               'SUELDO_ESTIMADO',
               'ANTIGUEDAD_MES',
               'Linea_Renovado',
               'Ahorro',
               'Prestamo_vigente',
               'Promed_6Mdeuda',
               'Deuda_Cubierta',
               'MES',
               'CLIENTE']

# ── Entrenamiento ─────────────────────────────────────────────────────────────
K_FEATURES       = 3
TEST_SIZE        = 0.3
RANDOM_STATE     = 42
TECNICA_BALANCEO = ['undersampling', 'oversampling','smote']

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT= os.getenv("MLFLOW_EXPERIMENT", "renovacion_prestamo-sinhue")
MLFLOW_RUN_NAME   = os.getenv('PIPELINE_VERSION', 'run-local')
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ── Validar Modelo ─────────────────────────────────────────────────────────────
UMBRAL_MIN=float(os.getenv("UMBRAL_MIN","0.6"))
MODEL_NAME=(os.getenv("MODEL_NAME","RenovacionPrestamo"))

# ── features para predecir con el modelo ────────────────────────────────────────
FEATURES = [
    "Plazo_Renovado",
    "Nro_Entidades",
    "Dif_Entidades",
    "Meses_oferta",
    "EDAD",
    "Flag_LimProv",
    "Uso_Linea_LOG",
    "Uso_TrimLinea_LOG",
    "Saldo_Consumo_LOG",
    "SUELDO_ESTIMADO_LOG",
    "ANTIGUEDAD_MES_LOG",
    "Linea_Renovado_LOG",
    "Ahorro_LOG",
    "Prestamo_vigente_LOG",
    "Promed_6Mdeuda_LOG",
    "Deuda_Cubierta_LOG",
    "REGION_CALLAO",
    "REGION_CENTRO",
    "REGION_LIMA BALNEARIO",
    "REGION_LIMA CENTRO",
    "REGION_LIMA ESTE",
    "REGION_LIMA MODERNA",
    "REGION_LIMA NORTE",
    "REGION_LIMA PROVINCIA",
    "REGION_LIMA SUR",
    "REGION_NORTE",
    "REGION_OESTE",
    "REGION_ORIENTE",
    "REGION_SIERRA CENTRAL",
    "REGION_SUR",
    "SEXO_F",
    "SEXO_M",
    "EST_CIVIL_C",
    "EST_CIVIL_D",
    "EST_CIVIL_S",
    "EST_CIVIL_U",
    "EST_CIVIL_V",
    "EST_CIVIL_X",
    "EST_CIVIL_Y",
    "Cluster"
]

# ── features para reporte Evidently ────────────────────────────────────────
NUMERICAL_FEATURES = [
    "Plazo_Renovado",
    "Nro_Entidades",
    "Dif_Entidades",
    "Meses_oferta",
    "EDAD",
    "Uso_Linea_LOG",
    "Uso_TrimLinea_LOG",
    "Saldo_Consumo_LOG",
    "SUELDO_ESTIMADO_LOG",
    "ANTIGUEDAD_MES_LOG",
    "Linea_Renovado_LOG",
    "Ahorro_LOG",
    "Prestamo_vigente_LOG",
    "Promed_6Mdeuda_LOG",
    "Deuda_Cubierta_LOG"
]

CATEGORICAL_FEATURES = [
    "Flag_LimProv",
    "REGION_CALLAO",
    "REGION_CENTRO",
    "REGION_LIMA BALNEARIO",
    "REGION_LIMA CENTRO",
    "REGION_LIMA ESTE",
    "REGION_LIMA MODERNA",
    "REGION_LIMA NORTE",
    "REGION_LIMA PROVINCIA",
    "REGION_LIMA SUR",
    "REGION_NORTE",
    "REGION_OESTE",
    "REGION_ORIENTE",
    "REGION_SIERRA CENTRAL",
    "REGION_SUR",
    "SEXO_F",
    "SEXO_M",
    "EST_CIVIL_C",
    "EST_CIVIL_D",
    "EST_CIVIL_S",
    "EST_CIVIL_U",
    "EST_CIVIL_V",
    "EST_CIVIL_X",
    "EST_CIVIL_Y",
    "Cluster"
]