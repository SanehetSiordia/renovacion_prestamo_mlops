"""
report_drift.py — genera un informe de drift de datos con Evidently.
Entrada :
    data/processed/processed_renovacion_prestamo.csv
Salida  : Reporte HTML en la ruta reports/reporte_drift_renovacion.html
  reporte_drift_renovacion.html     — drift en todas las features y target
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
import pandas as pd

from sklearn.model_selection import train_test_split
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import *
from evidently.renderers.html_widgets import WidgetSize

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

def generar_reporte_drift(df_reference: pd.DataFrame, df_current: pd.DataFrame) -> Report:
    column_mapping = ColumnMapping(
        target=C.TARGET,
        numerical_features=C.NUMERICAL_FEATURES,
        categorical_features=C.CATEGORICAL_FEATURES
    )

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    log.info("Ejecutando pruebas estadísticas de Data Drift y Target Drift...")

    report.run(
        reference_data=df_reference,
        current_data=df_current,
        column_mapping=column_mapping
    )

    report.save_html(str(C.REPORT_DRIFT_PATH))
    log.info(f'Reporte HTML guardado exitosamente en: {C.REPORT_DRIFT_PATH}')

    report.save_json(str(C.JSON_DRIFT_PATH))
    log.info(f'Reporte JSON guardado exitosamente en: {C.JSON_DRIFT_PATH}')
    return report

def generar_monitoreo_evidently(report: Report):
    log.info(f"Conectando al workspace de Evidently en: {C.EVIDENTLY_WORKSPACE_DIR}")
    ws = Workspace.create(str(C.EVIDENTLY_WORKSPACE_DIR))

    project_name = "Monitoreo Renovacion Prestamo"
    projects = ws.search_project(project_name)

    if not projects:
        project = ws.create_project(project_name)
        project.description = "Monitoreo de Data Drift y Target Drift para Renovación de Prestamos"

        project.dashboard.add_panel(
            DashboardPanelCounter(
                filter=ReportFilter(metadata_values={}, tag_values=[]),
                agg="last",
                title="Número de Features Evaluadas",
                value=PanelValue(
                    metric_id="DatasetDriftMetric",
                    field_path="number_of_columns",
                    legend="Features"
                ),
                size=WidgetSize.HALF,
            )
        )
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Proporción de Drift en Features",
                filter=ReportFilter(metadata_values={}, tag_values=[]),
                values=[
                    PanelValue(
                        metric_id="DatasetDriftMetric",
                        field_path="share_of_drifted_columns",
                        legend="Share of Drifted Features",
                    )
                ],
                plot_type=PlotType.LINE,
                size=WidgetSize.FULL,
            )
        )
        project.save()
        log.info(f"Proyecto nuevo '{project_name}' creado con paneles de dashboard.")
    else:
        project = projects[0]
        log.info(f"Proyecto existente encontrado: {project.name} (ID: {project.id})")

    ws.add_report(project.id, report)
    log.info(f"Reporte registrado exitosamente en el proyecto de Evidently UI.")
    return

# ── Función principal ─────────────────────────────────────────────────────
def run():
    log.info('=== ETAPA [1/3]: CARGAR DATOS PARA REPORTE ===')
    df_reference, df_current = cargar_datos()
    
    log.info("\n=== ETAPA [2/3] Generando reporte de Data & Target Drift ===")
    report = generar_reporte_drift(df_reference, df_current)

    log.info("\n=== ETAPA [3/3] Generando Monitero para visualización del reporte ===")
    generar_monitoreo_evidently(report)

    log.info('=== PROCESO FINALIZADO CON ÉXITO ===')

if __name__ == '__main__':
    run()