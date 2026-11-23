# Ruta de esquemas
import copy

from fastapi import APIRouter, HTTPException

from backend.config import TAREAS
from backend.servicios import cargador_modelos as cargador
from backend.servicios.recursos import cargar_esquema, referencias
from backend.servicios.validador import campos_planos

router = APIRouter(prefix="/api", tags=["esquemas"])


# Construye el esquema completo
def esquema_completo(tarea: str) -> dict:
    esquema = copy.deepcopy(cargar_esquema(tarea))
    ref = referencias()
    try:
        art = cargador.obtener(tarea)
        categorias, meta = art.categorias, art.metadata
    except RuntimeError:
        categorias, meta = {}, {}

    for campo in campos_planos(esquema):
        if campo["tipo"] == "texto":
            campo["valores_validos"] = categorias.get(campo["nombre"], [])

    esquema["clases"] = [
        {"clase": clase, **info} for clase, info in ref["clases"][tarea].items()
    ]
    esquema["tipo_modelo"] = meta.get("tipo")
    esquema["metricas_test"] = meta.get("metricas_test")
    esquema["rango_fechas"] = ref["rango_fechas"].get(tarea)
    return esquema


@router.get("/esquemas/{tarea}")
# Obtiene el esquema de una tarea
def obtener_esquema(tarea: str):
    if tarea not in TAREAS:
        raise HTTPException(status_code=404, detail=f"Tarea desconocida: {tarea}")
    return esquema_completo(tarea)
