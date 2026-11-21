from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ORIGINAL = ROOT / "datos" / "originales"
DATA_PROCESSED = ROOT / "datos" / "procesados"
MODELS = ROOT / "modelos"
ARTIFACTS = MODELS / "artefactos"
RESULTS = ROOT / "resultados"
PLOTS = ROOT / "graficos"
WEIGHTS = MODELS / "pesos"                 # pesos del mejor modelo de cada red

SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

EPOCHS = 160
BATCH_SIZE = 128
PATIENCE = 18

CONFIGS = [
    {"layers": [16], "lr": 0.001, "activation": "relu", "dropout": 0.15, "l2": 1e-4},
    {"layers": [32, 16], "lr": 0.001, "activation": "relu", "dropout": 0.20, "l2": 1e-4},
    {"layers": [64, 32], "lr": 0.001, "activation": "relu", "dropout": 0.25, "l2": 1e-4},
    {"layers": [64, 32, 16], "lr": 0.0005, "activation": "relu", "dropout": 0.25, "l2": 1e-4},
    {"layers": [128, 64, 32], "lr": 0.0005, "activation": "relu", "dropout": 0.30, "l2": 1e-4},
    {"layers": [32, 16], "lr": 0.001, "activation": "elu", "dropout": 0.20, "l2": 1e-4},
    {"layers": [64, 32, 16], "lr": 0.0005, "activation": "elu", "dropout": 0.25, "l2": 5e-4},
]

TASKS = {
    "pedidos": {
        "file": "pedidos_reforzado_etiquetado.csv",
        "target": "etiqueta_nivel_demora",
        "kind": "multiclass",
        "momento_prediccion": "Al recibir el pedido, antes de conocer la entrega real.",
    },
    "resenas": {
        "file": "resenas_reforzado_etiquetado.csv",
        "target": "etiqueta_nivel_satisfaccion",
        "kind": "multiclass",
        "momento_prediccion": "Después de la entrega y antes de recibir calificación/comentario.",
    },
    "precios": {
        "file": "historial_precios_reforzado_etiquetado.csv",
        "target": "etiqueta_cambio_fuerte",
        "kind": "binary",
        "momento_prediccion": "Antes de aplicar el nuevo precio.",
    },
}

# Columnas que NUNCA deben entrar como característica (fuga de la etiqueta o de información
# posterior al momento de predicción). El preprocesamiento las elimina por tarea.
COLUMNAS_PROHIBIDAS = {
    "pedidos": [
        "tiempo_entrega_minutos", "tiempo_entrega_orig", "calificacion_operativa",
        "estado_pedido", "estado_completado", "estado_es_incidencia",
        "hubo_demora", "etiqueta_hubo_demora", "etiqueta_nivel_demora",
        "dow_sin", "dow_cos",  # duplicadas: add_date_features las recrea como dia_semana_sin/cos
    ],
    "resenas": [
        "etiqueta_nivel_satisfaccion", "calificacion", "calificacion_orig",
        "comentario", "tiene_comentario", "nivel_rating_restaurante",
        "calificacion_operativa", "estado_pedido",
        "dow_sin", "dow_cos",
    ],
    "precios": [
        "etiqueta_cambio_fuerte", "precio_nuevo", "monto_cambio", "porcentaje_cambio",
        "pct_orig",  # porcentaje de cambio del dataset original (fuga directa)
    ],
}



def plots_dir(task_name):
    """Carpeta de gráficos de una red: graficos/<tarea>/"""
    return PLOTS / task_name


def results_dir(task_name):
    """Carpeta de resultados de una red: resultados/<tarea>/"""
    return RESULTS / task_name


def correlations_dir(task_name):
    """Matrices de correlación de una red: resultados/<tarea>/correlaciones/"""
    return results_dir(task_name) / "correlaciones"


def weights_dir(task_name):
    """Carpeta de pesos del mejor modelo de una red: modelos/pesos/<tarea>/"""
    return WEIGHTS / task_name


for folder in [DATA_PROCESSED, MODELS, ARTIFACTS, RESULTS, PLOTS, WEIGHTS]:
    folder.mkdir(parents=True, exist_ok=True)
for _task in TASKS:
    plots_dir(_task).mkdir(parents=True, exist_ok=True)
    results_dir(_task).mkdir(parents=True, exist_ok=True)
    correlations_dir(_task).mkdir(parents=True, exist_ok=True)
    weights_dir(_task).mkdir(parents=True, exist_ok=True)
