# Calcula variables derivadas para las predicciones
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from backend.config import RAIZ  # noqa: F401  (ruta del proyecto)
from backend.servicios.validador import campos_planos, clave_texto
from preprocesamiento.motor_preprocesamiento import add_date_features

NAN = float("nan")

# Cortes de demora
CORTES_DEMORA = (42, 51, 60)


@dataclass
# Contiene una fila construida
class FilaConstruida:
    df: pd.DataFrame
    avisos: list = field(default_factory=list)
    imputadas_previas: list = field(default_factory=list)  # datos previos


# Funciones comunes
# Obtiene datos del calendario
def _calendario(f: date) -> dict:
    # Devuelve columnas de calendario
    return {
        "dia": f.day,                                     # dia del mes
        "es_fin_de_semana": int(f.weekday() >= 5),
        "semana_anio": int(f.strftime("%U")) + 1,         # semana del ano
        "mes": f.month,
        "trimestre": (f.month - 1) // 3 + 1,
        "anio": f.year,
    }


# Agrega variables de fecha
def _con_fechas(columnas: dict, f: date) -> pd.DataFrame:
    # Crea un DataFrame con variables de fecha
    df = pd.DataFrame([{**columnas, "fecha": f.isoformat()}])
    return add_date_features(df)


DESCONOCIDO = "Desconocido"   # categoria desconocida


# Obtiene datos del pais
def _pais(ref: dict, ciudad: str, pais_indicado=None, avisos: list | None = None) -> tuple[str, str, bool]:
    # Obtiene pais y codigo de una ciudad
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


# Compara dos paises
def _mismo_pais(pais_a, conocido_a, pais_b, conocido_b, misma_ciudad: bool):
    # Compara los paises
    if misma_ciudad:
        return 1
    if conocido_a and conocido_b:
        return int(clave_texto(pais_a) == clave_texto(pais_b))
    return NAN


# Convierte un valor a entero
def _entero(valor: bool) -> int:
    return int(bool(valor))


# Convierte un valor numerico
def _num(valor):
    # Convierte valores vacios a NaN
    return NAN if valor is None else valor


# PEDIDOS
# Construye una fila de pedidos
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
        # Cliente
        "ciudad_cliente": v["ciudad_cliente"], "pais_cliente": pais_c, "codigo_pais_cliente": cod_c,
        "segmento_cliente": v["segmento_cliente"],
        "metodo_pago_preferido_cliente": v["metodo_pago_preferido_cliente"],
        "cliente_activo": _entero(v["cliente_activo"]),
        # Restaurante
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
        # Pedido
        "metodo_pago": v["metodo_pago"],
        "monto_total": v["monto_total"], "tarifa_entrega": v["tarifa_entrega"],
        "distancia_km": dist, "multiplicador_hora_pico": mult,
        # Variables derivadas
        "tarifa_por_km": v["tarifa_entrega"] / dist,
        "monto_por_km": v["monto_total"] / dist,
        "log_distancia": float(np.log1p(dist)),
        "es_hora_pico": int(mult > 1.0),
        "dist_x_pico": dist * mult,
        "cliente_misma_ciudad": int(misma_ciudad),
        "cliente_mismo_pais": _mismo_pais(pais_c, ok_c, pais_r, ok_r, misma_ciudad),
        "antiguedad_cliente_dias": max(0, antiguedad),
        # Historial
        "hist_tiempo_restaurante": _num(v.get("hist_tiempo_restaurante")),
        "hist_n_pedidos_restaurante": _num(v.get("hist_n_pedidos_restaurante")),
        "hist_tiempo_ciudad": _num(v.get("hist_tiempo_ciudad")),
    }
    return FilaConstruida(_con_fechas(columnas, f), avisos)


# RESENAS
# Construye una fila de resenas
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
        # Tiempo esperado
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


# PRECIOS
# Construye una fila de precios
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
        # Historial de precios
        ultimo = media_previa = lag
    else:
        # Primer cambio
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
        "hist_pct_std_previa": NAN,   # valor inicial
        "mes_sin": float(np.sin(2 * np.pi * f.month / 12)),
        "mes_cos": float(np.cos(2 * np.pi * f.month / 12)),
    }
    return FilaConstruida(_con_fechas(columnas, f), avisos)


CONSTRUCTORES = {
    "pedidos": fila_pedidos,
    "resenas": fila_resenas,
    "precios": fila_precios,
}


# Construye una fila para la tarea
def construir_fila(tarea: str, valores: dict, ref: dict, medianas: dict) -> FilaConstruida:
    return CONSTRUCTORES[tarea](valores, ref, medianas)


# Resumen para la interfaz
# Convierte un valor a tipo Python
def _py(x):
    # Convierte valores a tipos JSON
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    return x


# Lista las variables calculadas
def variables_calculadas(df: pd.DataFrame, esquema: dict) -> list[dict]:
    # Devuelve variables calculadas
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
