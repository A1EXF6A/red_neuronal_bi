"""Guarda el modelo .keras, los artefactos .joblib y los metadatos de cada red."""
import sys
from pathlib import Path
import json
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import MODELS, ARTIFACTS, results_dir


def exportar_todo(datos, seleccion, evaluacion, task_name, extras=None):
    task = datos["task"]
    label_encoder = datos["label_encoder"]
    best_model = seleccion["best_model"]
    MODELS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    carpeta_resultados = results_dir(task_name)
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

    model_path = MODELS / f"modelo_{task_name}.keras"
    best_model.save(model_path)
    joblib.dump(datos["preprocessor"], ARTIFACTS / f"preprocesador_{task_name}.joblib")
    joblib.dump(label_encoder, ARTIFACTS / f"codificador_{task_name}.joblib")

    metadata = {
        "modelo": task_name,
        "archivo_csv": task["file"],
        "etiqueta": task["target"],
        "tipo": task["kind"],
        "momento_prediccion": task["momento_prediccion"],
        "split_cronologico": datos["bundle"]["split"],
        "clases": label_encoder.classes_.tolist(),
        "dimensiones_entrada": int(datos["X_train"].shape[1]),
        "columnas_antes_de_codificar": datos["X_train_df"].columns.tolist(),
        "configuracion_seleccionada": seleccion["best_config"],
        "epocas_mejor_experimento": seleccion["best_epochs"],
        "macro_f1_validacion": float(seleccion["best_macro_f1"]),
        "metricas_test": evaluacion["test_metrics"],
        "parametros_entrenables": int(best_model.count_params()),
        "activacion_salida": "softmax" if task["kind"] == "multiclass" else "sigmoid",
        "nota": "Preprocesador ajustado solo con train; conjunto de prueba temporal reservado hasta la evaluación final.",
    }
    # Campos nuevos (no alteran los existentes): rutas de correlaciones y pesos exportados.
    if extras:
        metadata.update(extras)

    (carpeta_resultados / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Mejor configuración: {seleccion['best_config']}")
    print(f"Macro-F1 validación: {seleccion['best_macro_f1']:.4f}")
    print(f"Métricas test: {evaluacion['test_metrics']}")
    print(f"Modelo guardado: {model_path}")
    return metadata
