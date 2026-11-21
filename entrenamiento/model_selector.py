"""Bucle de experimentos: entrena cada configuración y elige la mejor por Macro-F1 de validación."""
import sys
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import EPOCHS, BATCH_SIZE, PATIENCE, CONFIGS
from model_builder import build_model
from model_utils import predict_classes, class_weights, metrics


def entrenar_configuraciones(datos, task_name):
    """Entrena todas las configuraciones de CONFIGS.

    Devuelve: experiment_rows, best_model, best_config, best_macro_f1, best_epochs.
    El mejor modelo conserva los pesos de su mejor época (restore_best_weights=True).
    """
    task = datos["task"]
    X_train, X_val = datos["X_train"], datos["X_val"]
    y_train, y_val = datos["y_train"], datos["y_val"]
    label_encoder = datos["label_encoder"]

    output_dim = len(label_encoder.classes_) if task["kind"] == "multiclass" else 1
    weights = class_weights(y_train)
    experiment_rows = []
    best_model = None
    best_config = None
    best_macro_f1 = -1.0
    best_epochs = 0

    for index, config in enumerate(CONFIGS, start=1):
        tf.keras.backend.clear_session()
        model = build_model(X_train.shape[1], output_dim, task["kind"], config)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5
            ),
        ]
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=weights,
            callbacks=callbacks,
            verbose=0,
            shuffle=False,
        )
        val_predictions = predict_classes(model, X_val, task["kind"])
        scores = metrics(y_val, val_predictions)
        row = {
            "experimento": index,
            "capas_ocultas": str(config["layers"]),
            "cantidad_capas_ocultas": len(config["layers"]),
            "activacion": config["activation"],
            "learning_rate": config["lr"],
            "dropout": config["dropout"],
            "l2": config["l2"],
            "epocas": len(history.history["loss"]),
            "parametros": int(model.count_params()),
            **scores,
        }
        experiment_rows.append(row)
        print(
            f"{config['layers']} {config['activation']} lr={config['lr']} "
            f"dropout={config['dropout']} L2={config['l2']} | "
            f"val_macro_f1={scores['macro_f1']:.4f} | val_acc={scores['accuracy']:.4f}"
        )
        if scores["macro_f1"] > best_macro_f1:
            best_macro_f1 = scores["macro_f1"]
            best_config = config.copy()
            best_model = model
            best_epochs = len(history.history["loss"])

    return {
        "experiment_rows": experiment_rows,
        "best_model": best_model,
        "best_config": best_config,
        "best_macro_f1": best_macro_f1,
        "best_epochs": best_epochs,
        "output_dim": output_dim,
    }
