"""Preparación determinista de variables y partición temporal de cada dataset.

Importante: aquí NO se ajustan imputadores, escaladores ni OneHotEncoder.
Esos pasos aprenden parámetros y se ajustan posteriormente usando SOLO train.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import (
    DATA_ORIGINAL, DATA_PROCESSED, TASKS, SEED, TRAIN_RATIO, VALIDATION_RATIO, COLUMNAS_PROHIBIDAS,
)


def read_csv(name):
    path = DATA_ORIGINAL / name
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el CSV original: {path}")
    return pd.read_csv(path)


def add_date_features(df):
    df = df.copy()
    if "fecha" in df.columns:
        dt = pd.to_datetime(df["fecha"], errors="coerce")
        df["fecha_anio"] = dt.dt.year
        df["fecha_mes"] = dt.dt.month
        df["fecha_dia"] = dt.dt.day
        df["fecha_dia_semana"] = dt.dt.dayofweek
        df["fecha_fin_semana"] = (dt.dt.dayofweek >= 5).astype("float")
        df["mes_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        df["mes_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        df["dia_semana_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["dia_semana_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        df = df.drop(columns=["fecha"])
    return df


def prepare_orders():
    df = read_csv(TASKS["pedidos"]["file"])
    y = df[TASKS["pedidos"]["target"]].copy()
    X = df.drop(columns=[c for c in COLUMNAS_PROHIBIDAS["pedidos"] if c in df.columns])
    # Se conservan las variables derivadas del CSV reforzado: tarifa_por_km, dist_x_pico,
    # es_hora_pico, cliente_misma_ciudad, antiguedad_cliente_dias y los históricos
    # hist_tiempo_restaurante / hist_tiempo_ciudad (solo usan pedidos anteriores).
    return add_date_features(X), y, df["fecha"]


def prepare_reviews():
    # El CSV reforzado de reseñas ya trae unidas las variables del pedido (tiempo de entrega,
    # distancia, tarifa, hora pico, nivel de demora, histórico del restaurante), así que ya no
    # se necesita cruzar con pedidos. Todas están disponibles tras la entrega y antes de la reseña.
    reviews = read_csv(TASKS["resenas"]["file"])
    y = reviews[TASKS["resenas"]["target"]].copy()
    X = reviews.drop(columns=[c for c in COLUMNAS_PROHIBIDAS["resenas"] if c in reviews.columns])
    return add_date_features(X), y, reviews["fecha"]


def add_price_history(df):
    df = df.copy()
    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    keys = [c for c in ["item_menu_id", "restaurante_id"] if c in df.columns]
    if not keys:
        keys = ["item_menu_key", "restaurante_key"]

    # Se agregan eventos por fecha y entidad; shift(1) impide usar el evento actual.
    hist = df.groupby(keys + ["fecha_dt"], dropna=False).agg(
        hist_pct_media_evento=("porcentaje_cambio", "mean"),
        hist_eventos_dia=("porcentaje_cambio", "size"),
    ).reset_index().sort_values(keys + ["fecha_dt"])
    grouped = hist.groupby(keys, dropna=False, sort=False)
    hist["hist_eventos_previos"] = grouped.cumcount()
    hist["hist_pct_ultimo"] = grouped["hist_pct_media_evento"].shift(1)
    hist["hist_pct_media_previa"] = grouped["hist_pct_media_evento"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    hist["hist_pct_std_previa"] = grouped["hist_pct_media_evento"].transform(
        lambda s: s.shift(1).expanding().std()
    )

    df = df.merge(
        hist[keys + ["fecha_dt", "hist_eventos_previos", "hist_pct_ultimo",
                     "hist_pct_media_previa", "hist_pct_std_previa"]],
        on=keys + ["fecha_dt"], how="left",
    )
    df["hist_eventos_previos"] = df["hist_eventos_previos"].fillna(0)
    df = df.drop(columns=["fecha_dt"] + COLUMNAS_PROHIBIDAS["precios"], errors="ignore")
    return add_date_features(df)


def prepare_prices():
    df = read_csv(TASKS["precios"]["file"])
    y = df[TASKS["precios"]["target"]].copy()
    X = add_price_history(df.drop(columns=[TASKS["precios"]["target"]]))
    return X, y, df["fecha"]


def clean_features(X):
    X = X.copy()
    drop = []
    for col in X.columns:
        low = str(col).lower()
        if (
            low.endswith("_key") or low.endswith("_id") or "hash" in low
            or "alias" in low or low.startswith("nombre_")
            or low in {"anio_mes", "fecha_key", "cambio_precio_key", "resena_key", "pedido_key"}
        ):
            drop.append(col)
    X = X.drop(columns=drop, errors="ignore")

    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype("int8")

    for col in list(X.select_dtypes(include=["object", "string", "category"]).columns):
        if X[col].nunique(dropna=True) > 100:
            X = X.drop(columns=[col])
    return X


def chronological_split(X, y, dates):
    dt = pd.to_datetime(dates, errors="coerce")
    order = np.argsort(dt.fillna(pd.Timestamp.min).values, kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = pd.Series(np.asarray(y)[order]).reset_index(drop=True)
    dates_sorted = dt.iloc[order].reset_index(drop=True)

    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))
    if min(train_end, val_end - train_end, n - val_end) < 2:
        raise ValueError("No hay suficientes registros para dividir 70/15/15.")

    return (
        X.iloc[:train_end].reset_index(drop=True),
        X.iloc[train_end:val_end].reset_index(drop=True),
        X.iloc[val_end:].reset_index(drop=True),
        y.iloc[:train_end].reset_index(drop=True),
        y.iloc[train_end:val_end].reset_index(drop=True),
        y.iloc[val_end:].reset_index(drop=True),
        dates_sorted.iloc[:train_end].reset_index(drop=True),
        dates_sorted.iloc[train_end:val_end].reset_index(drop=True),
        dates_sorted.iloc[val_end:].reset_index(drop=True),
    )


def export_csv(output_dir, task_name, parts):
    """Guarda un único CSV con todas las particiones: fecha, particion, variables preparadas y etiqueta.

    - `particion` vale train / validation / test; las filas van en orden cronológico
      (train = fechas más antiguas, test = más recientes).
    - `fecha` es solo de referencia, no es una característica (las variables de fecha ya están derivadas).
    - La etiqueta es la última columna.
    Son los mismos datos de particiones.joblib, aún sin imputación, escalado ni One-Hot
    (esos pasos se ajustan en el entrenamiento solo con train).
    """
    target = TASKS[task_name]["target"]
    tables = []
    for name, (X_part, y_part, d_part) in parts.items():
        table = X_part.copy()
        table.insert(0, "fecha", d_part.values)
        table.insert(1, "particion", name)
        table[target] = y_part.values
        tables.append(table)
    full = pd.concat(tables, ignore_index=True)
    path = output_dir / "datos_procesados.csv"
    full.to_csv(path, index=False, encoding="utf-8")
    return path.name


def prepare_task(task_name):
    if task_name == "pedidos":
        X, y, dates = prepare_orders()
    elif task_name == "resenas":
        X, y, dates = prepare_reviews()
    elif task_name == "precios":
        X, y, dates = prepare_prices()
    else:
        raise ValueError(f"Tarea desconocida: {task_name}")

    X = clean_features(X)
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    dates = dates.loc[valid].reset_index(drop=True)

    splits = chronological_split(X, y, dates)
    X_train, X_val, X_test, y_train, y_val, y_test, d_train, d_val, d_test = splits

    # Validar que todas las clases estén presentes en entrenamiento.
    classes_train = set(y_train.astype(str).unique())
    classes_all = set(y.astype(str).unique())
    if classes_train != classes_all:
        missing = sorted(classes_all - classes_train)
        raise ValueError(
            f"{task_name}: el bloque cronológico de entrenamiento no contiene "
            f"todas las clases {missing}. Revisa la distribución temporal."
        )

    output_dir = DATA_PROCESSED / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "task": task_name,
        "X_train": X_train, "X_validation": X_val, "X_test": X_test,
        "y_train": y_train, "y_validation": y_val, "y_test": y_test,
        "dates_train": d_train, "dates_validation": d_val, "dates_test": d_test,
        "feature_columns": X_train.columns.tolist(),
        "classes": sorted(classes_all),
        "split": {"train": TRAIN_RATIO, "validation": VALIDATION_RATIO, "test": 1 - TRAIN_RATIO - VALIDATION_RATIO},
        "note": "Particiones cronológicas. Imputación, escalado y OneHot se ajustan después solo con X_train.",
    }
    joblib.dump(bundle, output_dir / "particiones.joblib", compress=3)

    csv_file = export_csv(output_dir, task_name, {
        "train": (X_train, y_train, d_train),
        "validation": (X_val, y_val, d_val),
        "test": (X_test, y_test, d_test),
    })

    summary = {
        "tarea": task_name,
        "registros": len(X),
        "caracteristicas_antes_de_codificar": len(X.columns),
        "train": len(X_train), "validation": len(X_val), "test": len(X_test),
        "clases_train": y_train.astype(str).value_counts().to_dict(),
        "clases_validation": y_val.astype(str).value_counts().to_dict(),
        "clases_test": y_test.astype(str).value_counts().to_dict(),
        "columnas": X.columns.tolist(),
        "archivo_csv": csv_file,
        "columna_etiqueta_csv": TASKS[task_name]["target"],
        "columna_particion_csv": "particion",
    }
    import json
    (output_dir / "resumen_preparacion.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"{task_name}: train={len(X_train):,}, validación={len(X_val):,}, "
        f"prueba={len(X_test):,}; características candidatas={len(X.columns)}"
    )
    print(f"{task_name}: CSV guardado en {output_dir / csv_file}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tarea", choices=["pedidos", "resenas", "precios", "todos"])
    args = parser.parse_args()
    names = ["pedidos", "resenas", "precios"] if args.tarea == "todos" else [args.tarea]
    for name in names:
        prepare_task(name)
