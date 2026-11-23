# API de prediccion
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import FRONTEND_DIST, ORIGENES_PERMITIDOS
from backend.rutas import esquemas, modelos, prediccion
from backend.servicios import cargador_modelos

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")


@asynccontextmanager
# Gestiona el ciclo de vida
async def ciclo_de_vida(_app: FastAPI):
    cargador_modelos.cargar_todo()   # carga modelos
    yield


app = FastAPI(title="Predicción GlobalBite", version="1.0", lifespan=ciclo_de_vida)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENES_PERMITIDOS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(modelos.router)
app.include_router(esquemas.router)
app.include_router(prediccion.router)

# Servir interfaz compilada
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="interfaz")
