"""Matrices de correlación. Se calculan SOLO con train (sin usar validación ni test)."""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import correlations_dir
from graficacion import graficar_correlacion_variables, graficar_correlacion_etiqueta


def _limpiar_nombre(nombre):
    for prefijo in ("numeric__", "categorical__"):
        if nombre.startswith(prefijo):
            return nombre[len(prefijo):]
    return nombre


def generar_correlaciones(datos, task_name):
    """Genera 3 CSV (resultados/<tarea>/correlaciones/) y 2 gráficos (graficos/<tarea>/).

    1. Pearson entre variables numéricas originales (antes de One-Hot)  -> CSV + heatmap.
    2. Pearson entre todas las columnas que recibe la red (tras One-Hot) -> solo CSV.
    3. Spearman de cada columna codificada con la etiqueta               -> CSV + barras (top 20).
    """
    carpeta = correlations_dir(task_name)
    carpeta.mkdir(parents=True, exist_ok=True)
    y_train = datos["y_train"]

    # 1) numéricas originales
    numericas = datos["X_train_df"].select_dtypes(include=["number"])
    with np.errstate(all="ignore"):
        corr_num = numericas.corr(method="pearson")
    ruta_num = carpeta / "correlacion_numericas.csv"
    corr_num.to_csv(ruta_num)

    # 2) columnas codificadas (entrada real de la red)
    nombres = [_limpiar_nombre(n) for n in datos["feature_names"]]
    codificadas = pd.DataFrame(datos["X_train"], columns=nombres)
    with np.errstate(all="ignore"):
        corr_cod = codificadas.corr(method="pearson")
    ruta_cod = carpeta / "correlacion_codificadas.csv"
    corr_cod.to_csv(ruta_cod)

    # 3) variable vs. etiqueta (la etiqueta codificada conserva el orden de las clases)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # columnas constantes: correlación indefinida (NaN)
        corr_et = codificadas.corrwith(pd.Series(y_train), method="spearman")
    tabla_et = (
        corr_et.rename("spearman_con_etiqueta").rename_axis("variable").reset_index()
        .assign(abs_correlacion=lambda t: t["spearman_con_etiqueta"].abs())
        .sort_values("abs_correlacion", ascending=False)
    )
    ruta_et = carpeta / "correlacion_con_etiqueta.csv"
    tabla_et.to_csv(ruta_et, index=False)

    graf_vars = graficar_correlacion_variables(corr_num, task_name)
    graf_et = graficar_correlacion_etiqueta(corr_et, task_name)

    print(
        f"Correlaciones guardadas: {len(numericas.columns)} variables numéricas, "
        f"{len(nombres)} columnas codificadas."
    )
    return {
        "csv": [str(p.relative_to(ROOT)) for p in (ruta_num, ruta_cod, ruta_et)],
        "graficos": [str(p.relative_to(ROOT)) for p in (graf_vars, graf_et)],
    }
