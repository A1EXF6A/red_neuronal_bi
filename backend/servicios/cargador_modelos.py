"""Carga única (al iniciar) de preprocesadores, codificadores, metadatos y modelos .keras.

Los .joblib y los metadatos se cargan sin TensorFlow; el modelo .keras se carga aparte,
así el backend puede servir los formularios aunque TensorFlow no esté disponible.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field

import joblib
import numpy as np

from backend.config import ARTEFACTOS, MODELOS, RESULTADOS, TAREAS
from backend.servicios.validador import clave_texto

log = logging.getLogger("backend.cargador")


@dataclass
class Artefactos:
    tarea: str
    preprocesador: object
    codificador: object
    metadata: dict
    columnas: list                      # columnas_antes_de_codificar (orden del entrenamiento)
    categorias: dict                    # columna -> [valores válidos]
    indice_categorias: dict             # columna -> {clave_normalizada: valor_canónico}
    medianas: dict                      # columna numérica -> mediana de train (imputación)
    modelo: object = None               # modelo keras (se carga aparte)
    error_modelo: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_artefactos: dict[str, Artefactos] = {}
_errores: dict[str, str] = {}


def _extraer_categorias(pre) -> dict:
    """Categorías que aprendió el OneHotEncoder con train, por columna."""
    try:
        _, _, columnas = next(t for t in pre.transformers_ if t[0] == "categorical")
        onehot = pre.named_transformers_["categorical"].named_steps["onehot"]
    except (StopIteration, KeyError):
        return {}
    return {col: [str(c) for c in cats] for col, cats in zip(columnas, onehot.categories_)}


def _extraer_medianas(pre) -> dict:
    try:
        _, _, columnas = next(t for t in pre.transformers_ if t[0] == "numeric")
        estadisticos = pre.named_transformers_["numeric"].named_steps["imputer"].statistics_
    except (StopIteration, KeyError):
        return {}
    medianas = {}
    for col, valor in zip(columnas, estadisticos):
        if np.isfinite(valor):
            medianas[col] = float(valor)
    return medianas


def cargar_artefactos(tarea: str) -> Artefactos:
    pre = joblib.load(ARTEFACTOS / f"preprocesador_{tarea}.joblib")
    cod = joblib.load(ARTEFACTOS / f"codificador_{tarea}.joblib")
    meta = json.loads((RESULTADOS / tarea / "metadata.json").read_text(encoding="utf-8"))
    categorias = _extraer_categorias(pre)
    indice = {col: {clave_texto(v): v for v in vals} for col, vals in categorias.items()}
    return Artefactos(
        tarea=tarea,
        preprocesador=pre,
        codificador=cod,
        metadata=meta,
        columnas=list(meta["columnas_antes_de_codificar"]),
        categorias=categorias,
        indice_categorias=indice,
        medianas=_extraer_medianas(pre),
    )


def _cargar_keras(art: Artefactos) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf  # import diferido: es lento y puede no estar instalado
        art.modelo = tf.keras.models.load_model(MODELOS / f"modelo_{art.tarea}.keras", compile=False)
        art.error_modelo = None
    except Exception as exc:  # noqa: BLE001
        art.modelo = None
        art.error_modelo = f"{type(exc).__name__}: {exc}"
        log.error("No se pudo cargar el modelo de %s: %s", art.tarea, art.error_modelo)


def cargar_todo() -> None:
    for tarea in TAREAS:
        try:
            art = cargar_artefactos(tarea)
            _cargar_keras(art)
            _artefactos[tarea] = art
            _errores.pop(tarea, None)
            log.info("Red '%s' lista (modelo %s)", tarea, "cargado" if art.modelo else "NO cargado")
        except Exception as exc:  # noqa: BLE001
            _errores[tarea] = f"{type(exc).__name__}: {exc}"
            log.error("No se pudieron cargar los artefactos de %s: %s", tarea, _errores[tarea])


def obtener(tarea: str) -> Artefactos:
    if tarea not in _artefactos:
        raise RuntimeError(_errores.get(tarea, f"La red '{tarea}' no está cargada."))
    return _artefactos[tarea]


def estado() -> dict:
    salida = {}
    for tarea in TAREAS:
        art = _artefactos.get(tarea)
        salida[tarea] = {
            "artefactos": art is not None,
            "modelo": bool(art and art.modelo is not None),
            "error": _errores.get(tarea) or (art.error_modelo if art else None),
        }
    return salida
