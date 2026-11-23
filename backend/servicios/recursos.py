# Carga esquemas y referencias
import json
from functools import lru_cache

from backend.config import ESQUEMAS, REFERENCIAS, TAREAS


@lru_cache(maxsize=None)
# Carga el esquema de una tarea
def cargar_esquema(tarea: str) -> dict:
    if tarea not in TAREAS:
        raise KeyError(tarea)
    return json.loads((ESQUEMAS / f"{tarea}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
# Carga las referencias
def referencias() -> dict:
    if not REFERENCIAS.exists():
        raise FileNotFoundError(
            f"Falta {REFERENCIAS}. Ejecuta: python -m backend.scripts.generar_referencias"
        )
    return json.loads(REFERENCIAS.read_text(encoding="utf-8"))
