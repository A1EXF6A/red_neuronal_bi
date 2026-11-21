"""Todos los gráficos. Cada red guarda los suyos en graficos/<tarea>/."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import plots_dir

MAX_ETIQUETAS_EJE = 60      # por encima de este número de filas/columnas no se rotulan los ejes
TOP_CORRELACION = 20        # variables mostradas en el gráfico de correlación con la etiqueta


def _destino(task_name, nombre):
    carpeta = plots_dir(task_name)
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / nombre


# ---------------------------------------------------------------- evaluación
def graficar_matriz_confusion(matrix, clases, task_name):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=clases, yticklabels=clases,
    )
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title(f"Matriz de confusión - {task_name} (prueba temporal)")
    plt.tight_layout()
    ruta = _destino(task_name, "matriz_confusion_test.png")
    plt.savefig(ruta, dpi=160)
    plt.close()
    return ruta


def graficar_comparacion_experimentos(experiment_rows, task_name):
    plt.figure(figsize=(10, 5))
    labels = [
        f"{row['capas_ocultas']}\n{row['activacion']}\nLR={row['learning_rate']}"
        for row in experiment_rows
    ]
    plt.bar(range(len(experiment_rows)), [row["macro_f1"] for row in experiment_rows])
    plt.xticks(range(len(experiment_rows)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Macro-F1 de validación")
    plt.title(f"Comparación de experimentos - {task_name}")
    plt.tight_layout()
    ruta = _destino(task_name, "comparacion_experimentos.png")
    plt.savefig(ruta, dpi=160)
    plt.close()
    return ruta


def guardar_graficos(datos, seleccion, evaluacion, task_name):
    """Gráficos de evaluación (matriz de confusión y comparación de experimentos)."""
    return [
        graficar_matriz_confusion(evaluacion["matrix"], datos["label_encoder"].classes_, task_name),
        graficar_comparacion_experimentos(seleccion["experiment_rows"], task_name),
    ]


# -------------------------------------------------------------- correlaciones
def graficar_correlacion_variables(corr, task_name):
    """Heatmap (triángulo inferior) de la correlación entre variables numéricas originales."""
    n = corr.shape[0]
    lado = min(max(7, 0.38 * n), 22)
    mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    plt.figure(figsize=(lado + 1.5, lado))
    sns.heatmap(
        corr, mask=mascara, cmap="coolwarm", vmin=-1, vmax=1, center=0,
        annot=n <= 20, fmt=".2f", annot_kws={"size": 7},
        square=True, linewidths=0.2, cbar_kws={"shrink": 0.7},
    )
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(f"Correlación entre variables numéricas (train) - {task_name}")
    plt.tight_layout()
    ruta = _destino(task_name, "correlacion_variables.png")
    plt.savefig(ruta, dpi=160)
    plt.close()
    return ruta


def graficar_correlacion_etiqueta(serie, task_name, top=TOP_CORRELACION):
    """Barras horizontales con las variables más correlacionadas (Spearman) con la etiqueta."""
    serie = serie.dropna()
    seleccion = serie.reindex(serie.abs().sort_values(ascending=False).index)[:top][::-1]
    plt.figure(figsize=(8, max(4, 0.32 * len(seleccion) + 1.5)))
    colores = ["#c0392b" if v < 0 else "#2e86c1" for v in seleccion.values]
    plt.barh(seleccion.index, seleccion.values, color=colores)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Correlación de Spearman con la etiqueta (train)")
    plt.title(f"Top {len(seleccion)} variables vs. etiqueta - {task_name}")
    plt.yticks(fontsize=8)
    plt.tight_layout()
    ruta = _destino(task_name, "correlacion_con_etiqueta.png")
    plt.savefig(ruta, dpi=160)
    plt.close()
    return ruta


# ---------------------------------------------------------------------- pesos
def limpiar_graficos_pesos(task_name):
    """Borra heatmaps de pesos de ejecuciones anteriores (la cantidad de capas puede cambiar)."""
    for archivo in plots_dir(task_name).glob("pesos_capa_oculta_*.png"):
        archivo.unlink()


def graficar_pesos(matrices, task_name, nombres_entrada=None):
    """Un heatmap por cada matriz W de capa oculta del mejor modelo."""
    limpiar_graficos_pesos(task_name)
    rutas = []
    for i, W in enumerate(matrices, start=1):
        filas, cols = W.shape
        limite = float(np.max(np.abs(W))) or 1.0
        ancho = min(max(6, 0.25 * cols + 3), 20)
        alto = min(max(4, 0.12 * filas + 2), 20)
        plt.figure(figsize=(ancho, alto))
        sns.heatmap(
            W, cmap="coolwarm", vmin=-limite, vmax=limite, center=0,
            xticklabels=cols <= MAX_ETIQUETAS_EJE,
            yticklabels=(nombres_entrada if (i == 1 and nombres_entrada and filas <= MAX_ETIQUETAS_EJE)
                         else filas <= MAX_ETIQUETAS_EJE),
            cbar_kws={"shrink": 0.8},
        )
        plt.xlabel(f"Neuronas de la capa oculta {i}")
        plt.ylabel("Entradas" if i == 1 else f"Neuronas de la capa oculta {i - 1}")
        plt.title(f"Pesos W - capa oculta {i} ({filas}x{cols}) - {task_name}")
        plt.tight_layout()
        ruta = _destino(task_name, f"pesos_capa_oculta_{i:02d}.png")
        plt.savefig(ruta, dpi=160)
        plt.close()
        rutas.append(ruta)
    return rutas
