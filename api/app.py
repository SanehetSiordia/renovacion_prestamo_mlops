import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ClienteInput
from api.schemas import PrediccionOutput
from api.schemas import HealthResponse

from api.predictor import predictor

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | API | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Iniciando API en entorno: {os.getenv('ENV', 'dev')}")
    predictor.cargar()
    log.info(f"Modelo listo: {type(predictor.modelo).__name__}")
    yield
    log.info("API cerrando")

app = FastAPI(lifespan=lifespan)
app.title = os.getenv("APP_NAME", "Renovacion_Prestamo")
app.version= os.getenv("APP_VERSION", "1.0.0")
app.description=("Modelo clasifiacion de usuarios para renovacion de prestamo bancario."
                 "Modelo entrenado con el pipeline MLOps del proyecto con enfoque en RECALL."
                 )
app.docs_url="/docs"
app.debug=os.getenv("DEBUG", "False").lower() == "true"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Info"])
def root():
    """Información básica de la API."""
    return {
        "api":     os.getenv("APP_NAME", "Renovacion_Prestamo"),
        "env":     os.getenv("ENV", "dev"),
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "docs":    "/docs",
        "health":  "/health",
    }

@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    if predictor.modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no instanciado")
        
    is_dummy = not predictor.inicializado
    
    return HealthResponse(
        status="awaiting_train" if is_dummy else "ok",
        modelo="DummyModel (Esperando Entrenamiento)" if is_dummy else type(predictor.modelo).__name__,
        version=os.getenv("APP_VERSION", "1.0.0"),
        recall=float(predictor.metricas.get('metricas_evaluacion', {}).get('recall', 0)),
        env=os.getenv("ENV", "dev"),
    )

@app.post("/predecir", response_model=PrediccionOutput, tags=["Prediccion"])
def predecir(cliente: ClienteInput):    
    try:
        datos = cliente.model_dump(by_alias=True)
        return PrediccionOutput(**predictor.predecir(datos))
    
    except Exception as e:
        log.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))