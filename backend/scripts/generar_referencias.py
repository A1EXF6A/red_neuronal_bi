"""Genera backend/referencias/referencias.json a partir de los datos del proyecto.

Contiene lo que el backend necesita y que no se puede deducir de un solo registro:
  - ciudad -> país y código de país (cada ciudad pertenece a un solo país)
  - fecha base y constantes de relleno de la red de precios
  - rangos numéricos vistos en train (solo para avisar de extrapolación)
  - significado de cada clase (cortes del README)

Uso (desde la raíz):  python -m backend.scripts.generar_referencias
"""
import json

import pandas as pd

from backend.config import DATOS_ORIGINALES, DATOS_PROCESADOS, REFERENCIAS, TAREAS

CLASES = {
    "pedidos": {
        "0": {"nombre": "Entrega rápida", "detalle": "hasta 42 min"},
        "1": {"nombre": "Entrega normal", "detalle": "43 a 51 min"},
        "2": {"nombre": "Entrega lenta", "detalle": "52 a 60 min"},
        "3": {"nombre": "Entrega muy lenta", "detalle": "más de 60 min"},
    },
    "resenas": {
        "0": {"nombre": "Satisfacción baja", "detalle": "calificación hasta 3.2"},
        "1": {"nombre": "Satisfacción media-baja", "detalle": "calificación de 3.2 a 3.7"},
        "2": {"nombre": "Satisfacción media-alta", "detalle": "calificación de 3.7 a 4.2"},
        "3": {"nombre": "Satisfacción alta", "detalle": "calificación mayor a 4.2"},
    },
    "precios": {
        "0": {"nombre": "Cambio moderado", "detalle": "el precio sube 9.1 % o menos"},
        "1": {"nombre": "Cambio fuerte", "detalle": "el precio sube más de 9.1 %"},
    },
}


def main() -> None:
    pedidos = pd.read_csv(DATOS_ORIGINALES / "pedidos_reforzado_etiquetado.csv")
    resenas = pd.read_csv(DATOS_ORIGINALES / "resenas_reforzado_etiquetado.csv")
    precios = pd.read_csv(DATOS_ORIGINALES / "historial_precios_reforzado_etiquetado.csv",
                          parse_dates=["fecha"])

    # ciudad -> país / código
    partes = [
        pedidos[["ciudad_cliente", "pais_cliente", "codigo_pais_cliente"]]
        .set_axis(["ciudad", "pais", "codigo"], axis=1),
        pedidos[["ciudad_restaurante", "pais_restaurante", "codigo_pais_restaurante"]]
        .set_axis(["ciudad", "pais", "codigo"], axis=1),
    ]
    ciudades = pd.concat(partes).drop_duplicates()
    if ciudades["ciudad"].duplicated().any():
        raise ValueError("Hay ciudades asociadas a más de un país; revisa los datos.")
    mapa_ciudades = {
        r.ciudad: {"pais": r.pais, "codigo_pais": r.codigo} for r in ciudades.itertuples()
    }
    # las otras tablas no deben traer ciudades fuera del mapa
    for tabla, cols in [(resenas, ["ciudad_cliente", "ciudad_restaurante"]),
                        (precios, ["ciudad_restaurante"])]:
        for col in cols:
            desconocidas = set(tabla[col].unique()) - set(mapa_ciudades)
            if desconocidas:
                raise ValueError(f"Ciudades sin país en {col}: {sorted(desconocidas)}")

    sin_hist = precios[precios["n_cambio"] == 0].iloc[0]
    referencias = {
        "ciudades": dict(sorted(mapa_ciudades.items())),
        "fecha_base_precios": str(precios["fecha"].min().date()),
        "constantes_precios": {
            "lag_pct_sin_historial": float(sin_hist["lag_pct"]),
            "dias_desde_anterior_sin_historial": float(sin_hist["dias_desde_anterior"]),
        },
        "clases": CLASES,
        "rangos": {},
        "rango_fechas": {},
    }

    for tarea in TAREAS:
        proc = pd.read_csv(DATOS_PROCESADOS / tarea / "datos_procesados.csv", parse_dates=["fecha"])
        train = proc[proc["particion"] == "train"]
        numericas = train.select_dtypes("number").drop(
            columns=[c for c in train.columns if c.startswith("etiqueta_")], errors="ignore"
        )
        referencias["rangos"][tarea] = {
            col: [float(numericas[col].min()), float(numericas[col].max())]
            for col in numericas.columns
            if numericas[col].notna().any()
        }
        referencias["rango_fechas"][tarea] = [
            str(proc["fecha"].min().date()), str(proc["fecha"].max().date())
        ]

    REFERENCIAS.parent.mkdir(parents=True, exist_ok=True)
    REFERENCIAS.write_text(json.dumps(referencias, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Referencias guardadas en {REFERENCIAS}")


if __name__ == "__main__":
    main()
