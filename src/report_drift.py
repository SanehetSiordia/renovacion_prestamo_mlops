"""
report_drift.py — Genera informe de drift de datos y configura dashboard en Evidently 0.7.21.
Entrada : data/processed/processed_renovacion_prestamo.csv
Salida  : Reporte HTML, archivo JSON y Dashboard activo en Evidently UI Workspace.
"""

import logging
from pathlib import Path
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

logging.basicConfig(level=logging.INFO, format='%(asctime)s | REPORT_DRIFT | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

log = logging.getLogger(__name__)

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:

    if not C.PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset no encontrado: {C.PROCESSED_DATA_PATH}')
    
    df_processed = pd.read_csv(C.PROCESSED_DATA_PATH, sep=',')
    log.info(f'Cargado Dataframe: {df_processed.shape[0]} filas x {df_processed.shape[1]} columnas')

    df_reference, df_current = train_test_split(df_processed, test_size=0.3, random_state=42, stratify=df_processed[C.TARGET])
    log.info(f'Partición completada -> Referencia: {df_reference.shape[0]} registros | Actual: {df_current.shape[0]} registros')
    return df_reference, df_current

def generar_reporte_drift(df_reference: pd.DataFrame, df_current: pd.DataFrame):
    data_definition = DataDefinition(
        numerical_columns=C.NUMERICAL_FEATURES,
        categorical_columns=C.CATEGORICAL_FEATURES
    )

    reference_dataset = Dataset.from_pandas(df_reference, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(df_current, data_definition=data_definition)

    report = Report([DataDriftPreset()])
    log.info("Ejecutando pruebas estadísticas de Data Drift con Evidently 0.7.21...")

    report_result = report.run(
        current_data=current_dataset,
        reference_data=reference_dataset
    )

    report_result.save_html(str(C.REPORT_DRIFT_PATH))
    log.info(f'Reporte HTML guardado en: {C.REPORT_DRIFT_PATH}')
    
    report_result.save_json(str(C.JSON_DRIFT_PATH))
    log.info(f'Reporte JSON guardado en: {C.JSON_DRIFT_PATH}')

    return report_result

def generar_monitoreo_evidently(report_result):
    log.info(f"Conectando al workspace de Evidently en: {C.EVIDENTLY_WORKSPACE_DIR}")
    ws = Workspace.create(str(C.EVIDENTLY_WORKSPACE_DIR))

    project_name = "Monitoreo Renovacion Prestamo"
    projects = ws.search_project(project_name)

    if not projects:
        project = ws.create_project(project_name)
        project.description = "Monitoreo de Data Drift para Renovación de Préstamos"
        project.save()
        log.info(f"Proyecto nuevo '{project_name}' creado (ID: {project.id}).")
    else:
        project = projects[0]
        log.info(f"Proyecto existente encontrado: {project.name} (ID: {project.id})")

    ws.add_run(project.id, report_result)
    log.info("Reporte registrado exitosamente en el Workspace de Evidently UI.")

    log.info("Configurando paneles en el Dashboard de Evidently UI...")
    #Limpiar dashboard existente para evitar duplicados
    project.dashboard.clear_dashboard()

    tab_name = "Data Drift"

    # Panel encabezado de texto
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Dashboard de Monitoreo de Data Drift",
            size="full",
            values=[],
            plot_params={"plot_type": "text"},
        ),
        tab=tab_name,
    )

    # Panel 1: Contador - Columnas con Drift Detectado
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Columnas con Drift Detectado",
            subtitle="Cantidad de características con drift en la última evaluación",
            size="half",
            values=[
                PanelMetric(
                    legend="Columnas con Drift",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "count"},
                )
            ],
            plot_params={
                "plot_type": "counter",
                "aggregation": "last",
                "empty_value_text": "0 (Sin Drift)",   # Mensaje personalizado si no hay registros
                "default_value": "0",                  # Valor de respaldo numérico
            },
        ),
        tab=tab_name,
    )

    # Panel 2: Contador - Proporción Total de Drift
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Proporción Total de Drift",
            subtitle="Porcentaje de características con drift en la última evaluación",
            size="half",
            values=[
                PanelMetric(
                    legend="% Columnas Drift",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                )
            ],
            plot_params={
                "plot_type": "counter",
                "aggregation": "last",
                "empty_value_text": "0.0% (Sin Drift)", # Mensaje personalizado si no hay registros
                "default_value": "0.0%",
            },
        ),
        tab=tab_name,
    )

    # Panel 3: Gráfico de Línea - Evolución del share de Drift
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Tendencia de Proporción de Drift en Features",
            subtitle="Evolución temporal del share de drift en los reportes evaluados",
            size="full",
            values=[
                PanelMetric(
                    legend="Share of Drifted Columns",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                )
            ],
            plot_params={
                "plot_type": "line",
                "empty_value_text": "Sin historial de drift",
            },
        ),
        tab=tab_name,
    )

    # Guardar configuración del dashboard
    project.save()

    # Registrar el reporte / snapshot en el proyecto
    ws.add_run(project.id, report_result)
    log.info("Reporte registrado exitosamente en el Workspace de Evidently UI.")
    log.info(f"Dashboard actualizado y guardado exitosamente para el proyecto '{project_name}'.")
    return

# ── Función principal ─────────────────────────────────────────────────────
def run():
    log.info('=== ETAPA [1/3]: CARGAR DATOS PARA REPORTE ===')
    df_reference, df_current = cargar_datos()
    
    log.info("\n=== ETAPA [2/3] Generando reporte de Data & Target Drift ===")
    report_result = generar_reporte_drift(df_reference, df_current)

    log.info("\n=== ETAPA [3/3] Generando Monitero para visualización del reporte ===")
    generar_monitoreo_evidently(report_result)

    log.info('=== PROCESO FINALIZADO CON ÉXITO ===')

if __name__ == '__main__':
    run()