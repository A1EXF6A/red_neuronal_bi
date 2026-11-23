# Exporta pesos del mejor modelo
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import weights_dir
from graficacion import graficar_pesos


# Obtiene el nombre de una capa
def _nombre_clase(capa):
    return capa.__class__.__name__


# Extrae bloques del modelo
def _extraer_bloques(model):
    # Obtiene pesos de las capas
    capas = list(model.layers)
    indices_dense = [i for i, c in enumerate(capas) if _nombre_clase(c) == "Dense"]
    if not indices_dense:
        raise ValueError("El modelo no tiene capas Dense.")
    indice_salida = indices_dense[-1]

    ocultas = []
    for i in indices_dense[:-1]:
        W, b = capas[i].get_weights()[:2]
        bloque = {"W": W, "b": b, "units": int(W.shape[1]), "bn": None}
        siguiente = capas[i + 1] if i + 1 < len(capas) else None
        if siguiente is not None and _nombre_clase(siguiente) == "BatchNormalization":
            gamma, beta, media, varianza = siguiente.get_weights()
            bloque["bn"] = {
                "gamma": gamma, "beta": beta,
                "media_movil": media, "varianza_movil": varianza,
                "epsilon": float(siguiente.epsilon),
            }
        ocultas.append(bloque)

    W, b = capas[indice_salida].get_weights()[:2]
    salida = {"W": W, "b": b, "units": int(W.shape[1])}
    return ocultas, salida


# Limpia una carpeta
def _limpiar_carpeta(carpeta):
    # Elimina archivos de capas antiguas
    for patron in ("capa_oculta_*", "capa_salida_*", "pesos_*"):
        for archivo in carpeta.glob(patron):
            archivo.unlink()


# Genera nombres de neuronas
def _nombres_neuronas(n, prefijo="n"):
    return [f"{prefijo}{j:02d}" for j in range(1, n + 1)]


# Exporta los pesos
def exportar_pesos(datos, seleccion, task_name):
    model = seleccion["best_model"]
    config = seleccion["best_config"]
    carpeta = weights_dir(task_name)
    carpeta.mkdir(parents=True, exist_ok=True)
    _limpiar_carpeta(carpeta)

    ocultas, salida = _extraer_bloques(model)
    entradas = datos["feature_names"]
    clases = [str(c) for c in datos["label_encoder"].classes_]

    archivos = []
    arrays_npz = {}
    json_ocultas = []

    for k, bloque in enumerate(ocultas, start=1):
        tag = f"capa_oculta_{k:02d}"
        filas = entradas if k == 1 else _nombres_neuronas(ocultas[k - 2]["units"], f"capa{k - 1}_n")
        cols = _nombres_neuronas(bloque["units"])

        pd.DataFrame(bloque["W"], index=filas, columns=cols).rename_axis("entrada").to_csv(carpeta / f"{tag}_W.csv")
        pd.DataFrame({"sesgo_b": bloque["b"]}, index=cols).rename_axis("neurona").to_csv(carpeta / f"{tag}_b.csv")
        archivos += [f"{tag}_W.csv", f"{tag}_b.csv"]
        arrays_npz[f"{tag}_W"] = bloque["W"]
        arrays_npz[f"{tag}_b"] = bloque["b"]

        entrada_json = {
            "nombre": tag,
            "neuronas": bloque["units"],
            "activacion": config["activation"],
            "dropout": config["dropout"],
            "l2": config["l2"],
            "W": bloque["W"].tolist(),
            "b": bloque["b"].tolist(),
        }
        if bloque["bn"] is not None:
            bn = bloque["bn"]
            pd.DataFrame({
                "gamma": bn["gamma"], "beta": bn["beta"],
                "media_movil": bn["media_movil"], "varianza_movil": bn["varianza_movil"],
            }, index=cols).rename_axis("neurona").to_csv(carpeta / f"{tag}_batchnorm.csv")
            archivos.append(f"{tag}_batchnorm.csv")
            for clave in ("gamma", "beta", "media_movil", "varianza_movil"):
                arrays_npz[f"{tag}_bn_{clave}"] = bn[clave]
            entrada_json["batchnorm"] = {
                "gamma": bn["gamma"].tolist(), "beta": bn["beta"].tolist(),
                "media_movil": bn["media_movil"].tolist(),
                "varianza_movil": bn["varianza_movil"].tolist(),
                "epsilon": bn["epsilon"],
            }
        json_ocultas.append(entrada_json)

    # Capa de salida
    cols_salida = clases if salida["units"] == len(clases) else _nombres_neuronas(salida["units"], "salida")
    filas_salida = _nombres_neuronas(ocultas[-1]["units"] if ocultas else len(entradas), "capa_prev_n")
    pd.DataFrame(salida["W"], index=filas_salida, columns=cols_salida).rename_axis("entrada").to_csv(carpeta / "capa_salida_W.csv")
    pd.DataFrame({"sesgo_b": salida["b"]}, index=cols_salida).rename_axis("neurona").to_csv(carpeta / "capa_salida_b.csv")
    archivos += ["capa_salida_W.csv", "capa_salida_b.csv"]
    arrays_npz["capa_salida_W"] = salida["W"]
    arrays_npz["capa_salida_b"] = salida["b"]

    nombre_npz = f"pesos_{task_name}.npz"
    np.savez_compressed(carpeta / nombre_npz, **arrays_npz)

    tipo = datos["task"]["kind"]
    exportacion = {
        "modelo": task_name,
        "arquitectura": {
            "dimension_entrada": int(datos["X_train"].shape[1]),
            "capas_ocultas": config["layers"],
            "activacion": config["activation"],
            "learning_rate": config["lr"],
            "activacion_salida": "softmax" if tipo == "multiclass" else "sigmoid",
        },
        "columnas_entrada": entradas,
        "clases": clases,
        "capas_ocultas": json_ocultas,
        "capa_salida": {"W": salida["W"].tolist(), "b": salida["b"].tolist()},
        "nota": "Pesos del mejor modelo (mejor época). Orden de cada bloque: Dense -> BatchNorm -> Dropout "
                "(Dropout solo actúa en entrenamiento).",
    }
    nombre_json = f"pesos_{task_name}.json"
    (carpeta / nombre_json).write_text(json.dumps(exportacion, ensure_ascii=False), encoding="utf-8")
    archivos += [nombre_npz, nombre_json]

    graficos = graficar_pesos([b["W"] for b in ocultas], task_name, nombres_entrada=entradas)

    print(f"Pesos del mejor modelo guardados en: {carpeta} ({len(ocultas)} capas ocultas)")
    return {
        "carpeta": str(carpeta.relative_to(ROOT)),
        "capas_ocultas": len(ocultas),
        "formas_W_ocultas": [list(map(int, b["W"].shape)) for b in ocultas],
        "forma_W_salida": list(map(int, salida["W"].shape)),
        "archivos": sorted(archivos),
        "graficos": [str(p.relative_to(ROOT)) for p in graficos],
    }
