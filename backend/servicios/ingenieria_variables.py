"""Variables derivadas de cada red, calculadas con las MISMAS fórmulas del entrenamiento.

Origen de cada fórmula:
  - Fechas (fecha_*, mes_sin, dia_semana_sin, ...): se reutiliza `add_date_features`
    de preprocesamiento/motor_preprocesamiento.py (misma función que usó el entrenamiento).
  - Resto (tarifa_por_km, dist_x_pico, antiguedad_cliente_dias, ...): replicadas de
    generacion_datos/preparar_datos_reforzados.py y verificadas contra los CSV originales
    por backend/pruebas/test_paridad.py.

No depende de FastAPI ni de TensorFlow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from backend.config import RAIZ  # noqa: F401  (asegura sys.path hacia la raíz del proyecto)
from backend.servicios.validador import campos_planos, clave_texto
from preprocesamiento.motor_preprocesamiento import add_date_features

NAN = float("nan")

# Cortes de tiempo de entrega (min) que definen el nivel de demora: <=42, <=51, <=60, >60.
# El generador usa np.digitize(tiempo, [c + 0.5 for c in cortes]).
CORTES_DEMORA = (42, 51, 60)


@dataclass
class FilaConstruida:
    df: pd.DataFrame
    avisos: list = field(default_factory=list)
    imputadas_previas: list = field(default_factory=list)  # completadas antes del preprocesador


# ---------------------------------------------------------------- piezas comunes
def _calendario(f: date) -> dict:
    """Columnas de calendario que ya venían en los CSV (no las crea add_date_features)."""
    return {
        "dia": f.day,                                     # día del mes (1-31)
        "es_fin_de_semana": int(f.weekday() >= 5),
        "semana_anio": int(f.strftime("%U")) + 1,         # igual que en los datos (verificado 100 %)
        "mes": f.month,
        "trimestre": (f.month - 1) // 3 + 1,
        "anio": f.year,
    }


def _con_fechas(columnas: dict, f: date) -> pd.DataFrame:
    """Arma el DataFrame de una fila y añade las variables de fecha como en el entrenamiento."""
    df = pd.DataFrame([{**columnas, "fecha": f.isoformat()}])
    return add_date_features(df)


DESCONOCIDO = "Desconocido"   # categoría que el OneHotEncoder no vio -> todo ceros (se ignora)


def _pais(ref: dict, ciudad: str, pais_indicado=None, avisos: list | None = None) -> tuple[str, str, bool]:
    """(país, código, conocido). Funciona también con ciudades que no estaban en el entrenamiento.

    1) Ciudad conocida  -> país y código de las referencias.
    2) Ciudad nueva + país escrito por la persona -> ese país (código si el país es conocido).
    3) Ciudad nueva sin país -> «Desconocido» (el modelo lo ignora; no se inventa un país).
    """
    ciudades = {clave_texto(c): info for c, info in ref["ciudades"].items()}
    info = ciudades.get(clave_texto(ciudad))
    if info:
        if pais_indicado and clave_texto(pais_indicado) != clave_texto(info["pais"]) and avisos is not None:
            avisos.append(
                f"Se usó el país de «{ciudad}» ({info['pais']}) en lugar de «{pais_indicado}»."
            )
        return info["pais"], info["codigo_pais"], True
    if pais_indicado:
        codigos = {clave_texto(i["pais"]): i["codigo_pais"] for i in ref["ciudades"].values()}
        return str(pais_indicado), codigos.get(clave_texto(pais_indicado), DESCONOCIDO), True
    if avisos is not None:
        avisos.append(
            f"No indicaste el país de «{ciudad}»; el resultado puede ser menos preciso."
        )
    return DESCONOCIDO, DESCONOCIDO, False


def _mismo_pais(pais_a, conocido_a, pais_b, conocido_b, misma_ciudad: bool):
    """1/0 si se conocen ambos países; si no, NaN (el preprocesador usa la mediana de train)."""
    if misma_ciudad:
        return 1
    if conocido_a and conocido_b:
        return int(clave_texto(pais_a) == clave_texto(pais_b))
    return NAN


def _entero(valor: bool) -> int:
    return int(bool(valor))


def _num(valor):
    """None -> NaN (el preprocesador lo imputa con la mediana de train)."""
    return NAN if valor is None else valor


# ---------------------------------------------------------------- PEDIDOS
def fila_pedidos(v: dict, ref: dict, medianas: dict) -> FilaConstruida:
    f = v["fecha"]
    avisos = []
    pais_c, cod_c, ok_c = _pais(ref, v["ciudad_cliente"], v.get("pais_cliente"), avisos)
    pais_r, cod_r, ok_r = _pais(ref, v["ciudad_restaurante"], v.get("pais_restaurante"), avisos)
    misma_ciudad = clave_texto(v["ciudad_cliente"]) == clave_texto(v["ciudad_restaurante"])
    dist, mult = v["distancia_km"], v["multiplicador_hora_pico"]

    antiguedad = (f - v["fecha_registro_cliente"]).days
    if antiguedad < 0:
        avisos.append("La fecha de registro del cliente es posterior a la fecha del pedido; "
                      "se tomó la antigüedad como 0 días.")

    columnas = {
        **_calendario(f),
        # cliente
        "ciudad_cliente": v["ciudad_cliente"], "pais_cliente": pais_c, "codigo_pais_cliente": cod_c,
        "segmento_cliente": v["segmento_cliente"],
        "metodo_pago_preferido_cliente": v["metodo_pago_preferido_cliente"],
        "cliente_activo": _entero(v["cliente_activo"]),
        # restaurante
        "cocina_restaurante": v["cocina_restaurante"],
        "ciudad_restaurante": v["ciudad_restaurante"], "pais_restaurante": pais_r,
        "codigo_pais_restaurante": cod_r,
        "costo_promedio_dos_personas": v["costo_promedio_dos_personas"],
        "nivel_rating_restaurante": v["nivel_rating_restaurante"],
        "tiene_delivery": _entero(v["tiene_delivery"]),
        "tiene_comer_en_local": _entero(v["tiene_comer_en_local"]),
        "tiene_wifi": _entero(v["tiene_wifi"]),
        "acepta_tarjeta": _entero(v["acepta_tarjeta"]),
        "restaurante_activo": _entero(v["restaurante_activo"]),
        # pedido
        "metodo_pago": v["metodo_pago"],
        "monto_total": v["monto_total"], "tarifa_entrega": v["tarifa_entrega"],
        "distancia_km": dist, "multiplicador_hora_pico": mult,
        # derivadas del pedido
        "tarifa_por_km": v["tarifa_entrega"] / dist,
        "monto_por_km": v["monto_total"] / dist,
        "log_distancia": float(np.log1p(dist)),
        "es_hora_pico": int(mult > 1.0),
        "dist_x_pico": dist * mult,
        "cliente_misma_ciudad": int(misma_ciudad),
        "cliente_mismo_pais": _mismo_pais(pais_c, ok_c, pais_r, ok_r, misma_ciudad),
        "antiguedad_cliente_dias": max(0, antiguedad),
        # historial (opcional): NaN -> mediana de train en el preprocesador
        "hist_tiempo_restaurante": _num(v.get("hist_tiempo_restaurante")),
        "hist_n_pedidos_restaurante": _num(v.get("hist_n_pedidos_restaurante")),
        "hist_tiempo_ciudad": _num(v.get("hist_tiempo_ciudad")),
    }
    return FilaConstruida(_con_fechas(columnas, f), avisos)


# ---------------------------------------------------------------- RESEÑAS
def fila_resenas(v: dict, ref: dict, medianas: dict) -> FilaConstruida:
    f = v["fecha"]
    avisos = []
    pais_c, _, _ = _pais(ref, v["ciudad_cliente"], v.get("pais_cliente"), avisos)
    pais_r, _, _ = _pais(ref, v["ciudad_restaurante"], v.get("pais_restaurante"), avisos)
    tiempo = v["tiempo_entrega_minutos"]
    dist, mult = v["distancia_km"], v["multiplicador_hora_pico"]
    previas = []

    hist = v.get("hist_tiempo_restaurante")
    if hist is None:
        # tiempo_vs_esperado se calcula contra el histórico; si falta, se usa la mediana de train
        # (en el entrenamiento este campo nunca estuvo vacío).
        hist = float(medianas["hist_tiempo_restaurante"])
        previas.append({"nombre": "hist_tiempo_restaurante", "valor": round(hist, 6)})

    columnas = {
        **_calendario(f),
        "ciudad_cliente": v["ciudad_cliente"], "pais_cliente": pais_c,
        "segmento_cliente": v["segmento_cliente"],
        "cocina_restaurante": v["cocina_restaurante"],
        "ciudad_restaurante": v["ciudad_restaurante"], "pais_restaurante": pais_r,
        "tiempo_entrega_minutos": tiempo,
        "distancia_km": dist, "multiplicador_hora_pico": mult,
        "tarifa_entrega": v["tarifa_entrega"], "monto_total": v["monto_total"],
        "nivel_demora_pedido": int(np.digitize(tiempo, [c + 0.5 for c in CORTES_DEMORA])),
        "hist_tiempo_restaurante": hist,
        "tarifa_por_km": v["tarifa_entrega"] / dist,
        "tiempo_vs_esperado_restaurante": tiempo - hist,
        "es_hora_pico": int(mult > 1.0),
    }
    return FilaConstruida(_con_fechas(columnas, f), avisos, previas)


# ---------------------------------------------------------------- PRECIOS
def fila_precios(v: dict, ref: dict, medianas: dict) -> FilaConstruida:
    f = v["fecha"]
    base = date.fromisoformat(ref["fecha_base_precios"])
    const = ref["constantes_precios"]
    tiene_historial = bool(v["tiene_historial"])
    n_cambio = 1 if tiene_historial else 0
    avisos = []
    pais_r, _, _ = _pais(ref, v["ciudad_restaurante"], v.get("pais_restaurante"), avisos)

    if tiene_historial:
        lag = v["pct_cambio_anterior"]
        dias_anterior = v["dias_desde_anterior"]
        # Con un solo evento previo, "último" y "media previa" coinciden con ese cambio.
        ultimo = media_previa = lag
    else:
        # Primer cambio del ítem: mismos valores de relleno que usó el generador de datos.
        lag = const["lag_pct_sin_historial"]
        dias_anterior = const["dias_desde_anterior_sin_historial"]
        ultimo = media_previa = NAN

    columnas = {
        "mes": f.month, "trimestre": (f.month - 1) // 3 + 1, "anio": f.year,
        "categoria_item": v["categoria_item"], "cocina_item": v["cocina_item"],
        "disponible_item": _entero(v["disponible_item"]),
        "cocina_restaurante": v["cocina_restaurante"],
        "ciudad_restaurante": v["ciudad_restaurante"],
        "pais_restaurante": pais_r,
        "precio_anterior": v["precio_anterior"],
        "n_cambio": n_cambio,
        "dias_desde_inicio": (f - base).days,
        "dias_desde_anterior": dias_anterior,
        "lag_pct": lag,
        "tiene_historial": n_cambio,
        "log_precio_anterior": math.log(v["precio_anterior"]),
        "hist_pct_ciudad": _num(v.get("hist_pct_ciudad")),
        "hist_pct_categoria": _num(v.get("hist_pct_categoria")),
        "hist_eventos_previos": n_cambio,
        "hist_pct_ultimo": ultimo,
        "hist_pct_media_previa": media_previa,
        "hist_pct_std_previa": NAN,   # en train es siempre NaN; el imputador la ignora
        "mes_sin": float(np.sin(2 * np.pi * f.month / 12)),
        "mes_cos": float(np.cos(2 * np.pi * f.month / 12)),
    }
    return FilaConstruida(_con_fechas(columnas, f), avisos)


CONSTRUCTORES = {
    "pedidos": fila_pedidos,
    "resenas": fila_resenas,
    "precios": fila_precios,
}


def construir_fila(tarea: str, valores: dict, ref: dict, medianas: dict) -> FilaConstruida:
    return CONSTRUCTORES[tarea](valores, ref, medianas)


# ---------------------------------------------------------------- resumen para la interfaz
def _py(x):
    """numpy/pandas -> tipos JSON simples."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    return x


def variables_calculadas(df: pd.DataFrame, esquema: dict) -> list[dict]:
    """Columnas que el sistema derivó (no las escribió el usuario) y que tienen valor."""
    ingresadas = {c["nombre"] for c in campos_planos(esquema)}
    salida = []
    for col in df.columns:
        if col in ingresadas:
            continue
        valor = df.at[0, col]
        if isinstance(valor, float) and math.isnan(valor):
            continue
        salida.append({"nombre": col, "valor": _py(valor)})
    return salida
