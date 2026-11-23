# Rutas de modelos
from fastapi import APIRouter

from backend.config import TAREAS
from backend.servicios import cargador_modelos as cargador
from backend.servicios.recursos import cargar_esquema

router = APIRouter(prefix="/api", tags=["modelos"])


@router.get("/salud")
# Devuelve el estado del servicio
def salud():
    estado = cargador.estado()
    todo_listo = all(e["modelo"] for e in estado.values())
    return {"estado": "ok" if todo_listo else "incompleto", "redes": estado}


@router.get("/modelos")
# Lista los modelos disponibles
def listar_modelos():
    salida = []
    for tarea in TAREAS:
        esquema = cargar_esquema(tarea)
        item = {"tarea": tarea, "titulo": esquema["titulo"], "descripcion": esquema["descripcion"]}
        try:
            meta = cargador.obtener(tarea).metadata
            item.update(
                tipo=meta["tipo"],
                clases=meta["clases"],
                metricas_test=meta["metricas_test"],
                configuracion=meta["configuracion_seleccionada"],
                dimensiones_entrada=meta["dimensiones_entrada"],
            )
        except RuntimeError as exc:
            item["error"] = str(exc)
        salida.append(item)
    return salida
