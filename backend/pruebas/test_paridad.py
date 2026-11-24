# Prueba la paridad entre backend y entrenamiento
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from backend.config import DATOS_ORIGINALES
from backend.servicios.cargador_modelos import cargar_artefactos
from backend.servicios.ingenieria_variables import construir_fila
from backend.servicios.recursos import cargar_esquema, referencias
from backend.servicios.validador import validar
from preprocesamiento.motor_preprocesamiento import (
    clean_features, prepare_orders, prepare_prices, prepare_reviews,
)

MUESTRA = 400
SEMILLA = 0

# Tolerancias de paridad
TOLERANCIAS = {
    "precios": {"hist_pct_ultimo": 0.11, "hist_pct_media_previa": 0.11},
    "pedidos": {"hist_tiempo_ciudad": 10.0, "hist_tiempo_promedio": 15.0},
}


# Convierte un valor a booleano
def _bool(x) -> bool:
    return bool(int(x))


# Obtiene un valor opcional
def _opcional(fila, col):
    valor = fila[col]
    return None if pd.isna(valor) else valor


# Convierte una fila al formulario
def _formulario(tarea: str, f: pd.Series) -> dict:
    # Convierte una fila a datos del formulario
    fecha = str(f["fecha"])[:10]
    if tarea == "pedidos":
        return {
            "fecha": fecha, "monto_total": f["monto_total"], "tarifa_entrega": f["tarifa_entrega"],
            "distancia_km": f["distancia_km"], "multiplicador_hora_pico": f["multiplicador_hora_pico"],
            "metodo_pago": f["metodo_pago"],
            "metodo_pago_preferido_cliente": f["metodo_pago_preferido_cliente"],
            "ciudad_cliente": f["ciudad_cliente"], "segmento_cliente": f["segmento_cliente"],
            "fecha_registro_cliente": str(f["fecha_registro_cliente"])[:10],
            "cliente_activo": _bool(f["cliente_activo"]),
            "ciudad_restaurante": f["ciudad_restaurante"], "cocina_restaurante": f["cocina_restaurante"],
            "nivel_rating_restaurante": f["nivel_rating_restaurante"],
            "costo_promedio_dos_personas": f["costo_promedio_dos_personas"],
            "tiene_delivery": _bool(f["tiene_delivery"]),
            "tiene_comer_en_local": _bool(f["tiene_comer_en_local"]),
            "tiene_wifi": _bool(f["tiene_wifi"]), "acepta_tarjeta": _bool(f["acepta_tarjeta"]),
            "restaurante_activo": _bool(f["restaurante_activo"]),
            "misma_ubicacion": bool(f["ciudad_cliente"] == f["ciudad_restaurante"]),
            "hist_tiempo_promedio": f["hist_tiempo_restaurante"],
            "hist_n_pedidos_restaurante": int(f["hist_n_pedidos_restaurante"]),
        }
    if tarea == "resenas":
        return {
            "fecha": fecha, "tiempo_entrega_minutos": int(f["tiempo_entrega_minutos"]),
            "distancia_km": f["distancia_km"], "multiplicador_hora_pico": f["multiplicador_hora_pico"],
            "tarifa_entrega": f["tarifa_entrega"], "monto_total": f["monto_total"],
            "ciudad_cliente": f["ciudad_cliente"], "segmento_cliente": f["segmento_cliente"],
            "ciudad_restaurante": f["ciudad_restaurante"], "cocina_restaurante": f["cocina_restaurante"],
            "hist_tiempo_restaurante": f["hist_tiempo_restaurante"],
        }
    tiene_hist = int(f["n_cambio"]) > 0
    datos = {
        "fecha": fecha, "precio_anterior": f["precio_anterior"],
        "categoria_item": f["categoria_item"], "cocina_item": f["cocina_item"],
        "disponible_item": _bool(f["disponible_item"]),
        "cocina_restaurante": f["cocina_restaurante"], "ciudad_restaurante": f["ciudad_restaurante"],
        "tiene_historial": tiene_hist,
        "hist_pct_ciudad": f["hist_pct_ciudad"], "hist_pct_categoria": f["hist_pct_categoria"],
    }
    if tiene_hist:
        datos["pct_cambio_anterior"] = f["lag_pct"]
        datos["dias_desde_anterior"] = int(f["dias_desde_anterior"])
    return datos


# Obtiene datos de referencia
def _referencia(tarea: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Compara variables procesadas
    if tarea == "pedidos":
        X, _, _ = prepare_orders()
        original = pd.read_csv(DATOS_ORIGINALES / "pedidos_reforzado_etiquetado.csv")
    elif tarea == "resenas":
        X, _, _ = prepare_reviews()
        original = pd.read_csv(DATOS_ORIGINALES / "resenas_reforzado_etiquetado.csv")
    else:
        X, _, _ = prepare_prices()
        original = pd.read_csv(DATOS_ORIGINALES / "historial_precios_reforzado_etiquetado.csv")
    X = clean_features(X)
    assert len(X) == len(original), "El motor cambió el número de filas; no se puede alinear."
    return original, X


# Compara dos valores
def _iguales(a, b, tol: float) -> bool:
    a_nan = isinstance(a, float) and math.isnan(a)
    b_nan = isinstance(b, float) and math.isnan(b)
    if a_nan or b_nan:
        return a_nan and b_nan
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return abs(float(a) - float(b)) <= max(tol, 1e-6 * max(abs(float(a)), abs(float(b))))


# Comprueba la paridad
def comprobar(tarea: str) -> dict:
    art = cargar_artefactos(tarea)
    esquema, ref = cargar_esquema(tarea), referencias()
    original, X_ref = _referencia(tarea)
    tolerancias = TOLERANCIAS.get(tarea, {})

    rng = np.random.default_rng(SEMILLA)
    indices = rng.choice(len(original), size=min(MUESTRA, len(original)), replace=False)

    diferencias: list[str] = []
    filas_backend, filas_ref = [], []
    for i in indices:
        datos = _formulario(tarea, original.iloc[i])
        valores, errores = validar(esquema, datos, art.indice_categorias)
        if errores:
            diferencias.append(f"fila {i}: el validador rechazó datos reales: {errores}")
            continue
        df = construir_fila(tarea, valores, ref, art.medianas).df
        faltan = [c for c in art.columnas if c not in df.columns]
        if faltan:
            diferencias.append(f"fila {i}: faltan columnas {faltan}")
            continue
        df = df[art.columnas]
        esperado = X_ref.iloc[[i]][art.columnas].reset_index(drop=True)
        for col in art.columnas:
            if not _iguales(df.at[0, col], esperado.at[0, col], tolerancias.get(col, 0.0)):
                diferencias.append(f"fila {i} · {col}: backend={df.at[0, col]!r} "
                                   f"entrenamiento={esperado.at[0, col]!r}")
        filas_backend.append(df)
        filas_ref.append(esperado)

    # Entrada de la red
    max_dif = 0.0
    if filas_backend:
        A = art.preprocesador.transform(pd.concat(filas_backend, ignore_index=True))
        B = art.preprocesador.transform(pd.concat(filas_ref, ignore_index=True))
        assert A.shape[1] == art.metadata["dimensiones_entrada"]
        max_dif = float(np.max(np.abs(A - B)))

    return {"tarea": tarea, "filas": len(filas_backend), "columnas": len(art.columnas),
            "diferencias": diferencias, "max_dif_preprocesada": max_dif}


# Verifica una tarea
def _verificar(tarea: str) -> None:
    r = comprobar(tarea)
    assert not r["diferencias"], "\n".join(r["diferencias"][:15])
    # Variacion permitida
    assert r["max_dif_preprocesada"] < 5.0, f"Diferencia en la entrada de la red: {r['max_dif_preprocesada']}"


# Prueba la paridad de pedidos
def test_paridad_pedidos():
    _verificar("pedidos")


# Prueba la paridad de resenas
def test_paridad_resenas():
    _verificar("resenas")


# Prueba la paridad de precios
def test_paridad_precios():
    _verificar("precios")


if __name__ == "__main__":
    fallo = False
    for t in ("pedidos", "resenas", "precios"):
        r = comprobar(t)
        estado = "OK " if not r["diferencias"] else "FALLO"
        print(f"[{estado}] {t}: {r['filas']} filas · {r['columnas']} columnas comparadas · "
              f"dif. máx. tras preprocesador = {r['max_dif_preprocesada']:.6f}")
        for linea in r["diferencias"][:10]:
            print("    ", linea)
        fallo = fallo or bool(r["diferencias"])
    sys.exit(1 if fallo else 0)
