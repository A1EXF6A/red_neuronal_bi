"""Rutas y constantes del backend. Todas relativas a la raíz del proyecto."""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
RAIZ = BACKEND.parent

# Permite importar `preprocesamiento.motor_preprocesamiento` y `configuracion` del proyecto.
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

MODELOS = RAIZ / "modelos"
ARTEFACTOS = MODELOS / "artefactos"
RESULTADOS = RAIZ / "resultados"
DATOS_ORIGINALES = RAIZ / "datos" / "originales"
DATOS_PROCESADOS = RAIZ / "datos" / "procesados"

ESQUEMAS = BACKEND / "esquemas"
REFERENCIAS = BACKEND / "referencias" / "referencias.json"
FRONTEND_DIST = RAIZ / "frontend" / "dist"

TAREAS = ("pedidos", "resenas", "precios")

ORIGENES_PERMITIDOS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
