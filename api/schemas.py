"""api/schemas.py — Modelos Pydantic para validación de entrada y salida."""
from pydantic import Field
from pydantic import BaseModel
from typing import Literal

class ClienteInput(BaseModel):
    # --- VARIABLES ENTERAS (Conteos, Edades y Clústeres) ---------------------
    Plazo_Renovado: int = Field(..., description="Número de meses ofertados.")
    Nro_Entidades: int = Field(..., description="Número de entidades financieras del sistema financiero.")
    Dif_Entidades: int = Field(..., description="Diferencia de entidades financieras en el último año.")
    Meses_oferta: int = Field(..., description="Número de meses transcurridos desde la primera oferta.")
    EDAD: int = Field(..., description="Edad del cliente.")
    Flag_LimProv: int = Field(..., description="Flag si vive en Lima (0) o provincias (1).")
    Cluster: int = Field(..., description="Identificador del segmento o clúster generado por KMeans.")
    
    # --- VARIABLES FLOTANTES (Transformaciones LOG - Continuos) --------------
    Uso_Linea_LOG: float = Field(..., description="Logaritmo del porcentaje de uso de línea de TC del SF.")
    Uso_TrimLinea_LOG: float = Field(..., description="Logaritmo del porcentaje de uso de línea de TC del banco.")
    Saldo_Consumo_LOG: float = Field(..., description="Logaritmo del monto de saldo de consumo.")
    SUELDO_ESTIMADO_LOG: float = Field(..., description="Logaritmo del monto de sueldo estimado del cliente.")
    ANTIGUEDAD_MES_LOG: float = Field(..., description="Logaritmo del número de meses de antigüedad como cliente.")
    Linea_Renovado_LOG: float = Field(..., description="Logaritmo del monto de línea ofrecida para la renovación.")
    Ahorro_LOG: float = Field(..., description="Logaritmo del monto de ahorro en el banco.")
    Prestamo_vigente_LOG: float = Field(..., description="Logaritmo del monto de préstamo vigente en el banco.")
    Promed_6Mdeuda_LOG: float = Field(..., description="Logaritmo del monto del promedio semestral de deuda.")
    Deuda_Cubierta_LOG: float = Field(..., description="Logaritmo del porcentaje de deuda cubierta con la renovación.")

    # --- VARIABLES GEOGRÁFICAS (One-Hot Encoding - Valores: 0 o 1) -----------
    REGION_CALLAO: int = Field(..., description="Flag de residencia en la región Callao.")
    REGION_CENTRO: int = Field(..., description="Flag de residencia en la región Centro.")
    REGION_LIMA_BALNEARIO: int = Field(..., alias="REGION_LIMA BALNEARIO", description="Flag de residencia en Lima Balnearios.")
    REGION_LIMA_CENTRO: int = Field(..., alias="REGION_LIMA CENTRO", description="Flag de residencia en Lima Centro.")
    REGION_LIMA_ESTE: int = Field(..., alias="REGION_LIMA ESTE", description="Flag de residencia en Lima Este.")
    REGION_LIMA_MODERNA: int = Field(..., alias="REGION_LIMA MODERNA", description="Flag de residencia en Lima Moderna.")
    REGION_LIMA_NORTE: int = Field(..., alias="REGION_LIMA NORTE", description="Flag de residencia en Lima Norte.")
    REGION_LIMA_PROVINCIA: int = Field(..., alias="REGION_LIMA PROVINCIA", description="Flag de residencia en Lima Provincias.")
    REGION_LIMA_SUR: int = Field(..., alias="REGION_LIMA SUR", description="Flag de residencia en Lima Sur.")
    REGION_NORTE: int = Field(..., description="Flag de residencia en la región Norte.")
    REGION_OESTE: int = Field(..., description="Flag de residencia en la región OESTE.")
    REGION_ORIENTE: int = Field(..., description="Flag de residencia en la región Oriente.")
    REGION_SIERRA_CENTRAL: int = Field(..., alias="REGION_SIERRA CENTRAL", description="Flag de residencia en la región Sierra Central.")
    REGION_SUR: int = Field(..., description="Flag de residencia en la región Sur.")

    # --- VARIABLES DEMOGRÁFICAS (One-Hot Encoding - Valores: 0 o 1) ----------
    SEXO_F: int = Field(..., description="Flag para sexo Femenino.")
    SEXO_M: int = Field(..., description="Flag para sexo Masculino.")

    EST_CIVIL_C: int = Field(..., description="Flag para Estado Civil: Casado.")
    EST_CIVIL_D: int = Field(..., description="Flag para Estado Civil: Divorciado.")
    EST_CIVIL_S: int = Field(..., description="Flag para Estado Civil: Soltero.")
    EST_CIVIL_U: int = Field(..., description="Flag para Estado Civil: Unión Libre.")
    EST_CIVIL_V: int = Field(..., description="Flag para Estado Civil: Viudo.")
    EST_CIVIL_X: int = Field(..., description="Flag para Estado Civil: Tipo X.")
    EST_CIVIL_Y: int = Field(..., description="Flag para Estado Civil: Tipo Y.")

    # ── Configuración del Schema y Ejemplo de API ────────────────────────────
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Plazo_Renovado": 12,
                "Nro_Entidades": 3,
                "Dif_Entidades": 0,
                "Meses_oferta": 6,
                "EDAD": 34,
                "Flag_LimProv": 0,
                "Uso_Linea_LOG": 0.440676,
                "Uso_TrimLinea_LOG": 0.440676,
                "Saldo_Consumo_LOG": 8.88,
                "SUELDO_ESTIMADO_LOG": 8.134,
                "ANTIGUEDAD_MES_LOG": 5.0,
                "Linea_Renovado_LOG": 1500.0,
                "Ahorro_LOG": 2093.0,
                "Prestamo_vigente_LOG": 18.512,
                "Promed_6Mdeuda_LOG": 8.051,
                "Deuda_Cubierta_LOG": 0.67,
                "REGION_CALLAO": 0,
                "REGION_CENTRO": 1,
                "REGION_LIMA BALNEARIO": 0,
                "REGION_LIMA CENTRO": 0,
                "REGION_LIMA ESTE": 0,
                "REGION_LIMA MODERNA": 0,
                "REGION_LIMA NORTE": 0,
                "REGION_LIMA PROVINCIA": 0,
                "REGION_LIMA SUR": 0,
                "REGION_NORTE": 0,
                "REGION_OESTE": 0,
                "REGION_ORIENTE": 0,
                "REGION_SIERRA CENTRAL": 0,
                "REGION_SUR": 0,
                "SEXO_F": 0,
                "SEXO_M": 1,
                "EST_CIVIL_C": 1,
                "EST_CIVIL_D": 0,
                "EST_CIVIL_S": 0,
                "EST_CIVIL_U": 0,
                "EST_CIVIL_V": 0,
                "EST_CIVIL_X": 0,
                "EST_CIVIL_Y": 0,
                "Cluster": 1
            }
        }
    }

class PrediccionOutput(BaseModel):
    """Respuesta del clasificador para renovacion de prestamo."""
    score_riesgo:         float
    decision:             Literal["USUARIO PARA RENOVACION DETECTADO", "NORMAL"]
    probabilidad_pico:    float
    umbral_usado:         float
    modelo:               str


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""
    status:  str
    modelo:  str
    version: str
    recall:  float
    env:     str
