"""POST /api/predecir/{tarea}"""
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import TAREAS
from backend.servicios import predictor
from backend.servicios.validador import ErrorValidacion

router = APIRouter(prefix="/api", tags=["prediccion"])


class Solicitud(BaseModel):
    datos: dict[str, Any]


@router.post("/predecir/{tarea}")
def predecir(tarea: str, solicitud: Solicitud):
    if tarea not in TAREAS:
        raise HTTPException(status_code=404, detail=f"Tarea desconocida: {tarea}")
    try:
        return predictor.predecir(tarea, solicitud.datos)
    except ErrorValidacion as exc:
        raise HTTPException(status_code=422, detail={"errores": exc.errores})
    except predictor.ModeloNoDisponible as exc:
        raise HTTPException(status_code=503, detail={"mensaje": f"Modelo no disponible: {exc}"})
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail={"mensaje": str(exc)})
