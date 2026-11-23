# Coordina el proceso de prediccion
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import TAREAS
from backend.servicios import cargador_modelos as cargador
from backend.servicios.ingenieria_variables import construir_fila, variables_calculadas
from backend.servicios.recursos import cargar_esquema, referencias
from backend.servicios.validador import ErrorValidacion, avisos_categorias_nuevas, avisos_de_rango, validar


# Representa un modelo no disponible
class ModeloNoDisponible(Exception):
    pass


# Obtiene variables imputadas
def _imputadas(df: pd.DataFrame, art, previas: list) -> list[dict]:
    # Obtiene variables numericas sin valor
    resultado = list(previas)
    for col, mediana in art.medianas.items():
        if col in df.columns and pd.isna(df.at[0, col]):
            resultado.append({"nombre": col, "valor": round(mediana, 6)})
    return resultado


# Realiza una prediccion
def predecir(tarea: str, datos: dict) -> dict:
    if tarea not in TAREAS:
        raise KeyError(tarea)

    art = cargador.obtener(tarea)
    if art.modelo is None:
        raise ModeloNoDisponible(
            art.error_modelo or "El modelo .keras no se cargó (¿está instalado TensorFlow?)."
        )

    esquema, ref = cargar_esquema(tarea), referencias()

    # Validar datos
    valores, errores = validar(esquema, datos, art.indice_categorias)
    if errores:
        raise ErrorValidacion(errores)
    advertencias = avisos_de_rango(
        esquema, valores, ref["rangos"].get(tarea, {}), ref["rango_fechas"].get(tarea)
    )

    advertencias += avisos_categorias_nuevas(esquema, valores, art.indice_categorias)

    # Variables derivadas
    fila = construir_fila(tarea, valores, ref, art.medianas)
    advertencias += fila.avisos

    # Ordenar columnas
    faltan = [c for c in art.columnas if c not in fila.df.columns]
    if faltan:
        raise RuntimeError(f"Faltan columnas para el preprocesador: {faltan}")
    df = fila.df[art.columnas]

    # Preprocesar
    X = art.preprocesador.transform(df).astype("float32")
    esperado = int(art.metadata["dimensiones_entrada"])
    if X.shape[1] != esperado:
        raise RuntimeError(f"Dimensión de entrada {X.shape[1]} != {esperado} del entrenamiento")

    # Ejecutar modelo
    with art.lock:
        salida = np.asarray(art.modelo.predict(X, verbose=0), dtype="float64")

    # Obtener clase
    if salida.shape[-1] == 1:                       # binario con sigmoide
        p1 = float(salida[0, 0])
        probabilidades = [1.0 - p1, p1]
    else:                                           # multiclase con softmax
        probabilidades = [float(p) for p in salida[0]]
    clases = [str(c) for c in art.codificador.classes_]
    indice = int(np.argmax(probabilidades))
    clase = str(art.codificador.inverse_transform([indice])[0])

    significados = ref["clases"][tarea]
    detalle = [
        {
            "clase": c,
            "nombre": significados[c]["nombre"],
            "detalle": significados[c]["detalle"],
            "probabilidad": round(p, 6),
        }
        for c, p in zip(clases, probabilidades)
    ]
    if max(probabilidades) < 0.5:
        advertencias.append("Resultado poco claro: ninguna opción supera el 50 %.")

    return {
        "tarea": tarea,
        "clase": clase,
        "nombre": significados[clase]["nombre"],
        "detalle": significados[clase]["detalle"],
        "confianza": round(max(probabilidades), 6),
        "probabilidades": detalle,
        "variables_calculadas": variables_calculadas(df, esquema),
        "variables_imputadas": _imputadas(df, art, fila.imputadas_previas),
        "advertencias": advertencias,
    }
