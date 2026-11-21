"""Funciones auxiliares: predicción de clases, pesos de clase y métricas."""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def predict_classes(model, X, kind):
    probabilities = model.predict(X, verbose=0)
    if kind == "binary":
        return (probabilities.ravel() >= 0.5).astype(int)
    return np.argmax(probabilities, axis=1)


def class_weights(y):
    classes = np.unique(y)
    values = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, values)}


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
