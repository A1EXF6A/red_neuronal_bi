# Evalua el mejor modelo
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import results_dir
from model_utils import predict_classes, metrics


# Evalua el conjunto de prueba
def evaluar_test(datos, seleccion, task_name):
    task = datos["task"]
    label_encoder = datos["label_encoder"]
    y_test = datos["y_test"]
    carpeta = results_dir(task_name)
    carpeta.mkdir(parents=True, exist_ok=True)

    experiments = pd.DataFrame(seleccion["experiment_rows"]).sort_values("macro_f1", ascending=False)
    experiments.to_csv(carpeta / "experimentos.csv", index=False)

    # Evaluacion final
    test_predictions = predict_classes(seleccion["best_model"], datos["X_test"], task["kind"])
    test_metrics = metrics(y_test, test_predictions)
    class_ids = np.arange(len(label_encoder.classes_))
    matrix = confusion_matrix(y_test, test_predictions, labels=class_ids)

    pd.DataFrame(
        matrix, index=label_encoder.classes_, columns=label_encoder.classes_
    ).to_csv(carpeta / "matriz_confusion_test.csv")

    report = classification_report(
        y_test, test_predictions, labels=class_ids,
        target_names=[str(value) for value in label_encoder.classes_],
        output_dict=True, zero_division=0,
    )
    (carpeta / "reporte_test.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "test_predictions": test_predictions,
        "test_metrics": test_metrics,
        "matrix": matrix,
        "report": report,
    }
