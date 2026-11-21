# Visualizacion de clases: Historial de precios
# Instalar dependencias: pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ARCHIVO = "../originales/historial_precios_reforzado_etiquetado.csv"
ETIQUETA = "etiqueta_cambio_fuerte"
MAX_TSNE = 10000
RANDOM_STATE = 42

df = pd.read_csv(ARCHIVO)

# Excluir etiquetas, identificadores, fechas y texto descriptivo.
excluir_exactas = {
    ETIQUETA, "fecha", "nombre_mes", "nombre_item", "nombre_restaurante",
    "cocina_item", "cocina_restaurante", "ciudad_restaurante",
    "pais_restaurante", "anio_mes"
}
excluir = set(excluir_exactas)
for col in df.columns:
    if col.endswith("_key") or col.endswith("_id"):
        excluir.add(col)

X = df.drop(columns=[c for c in excluir if c in df.columns], errors="ignore")
y = df[ETIQUETA].astype(str)

# Usar variables numericas y categoricas de cardinalidad baja.
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.select_dtypes(include=["object", "category", "bool"]).columns
             if X[c].nunique(dropna=True) <= 30]

X = X[num_cols + cat_cols].copy()
X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
for c in cat_cols:
    X[c] = X[c].fillna("Desconocido").astype(str)

preprocesador = ColumnTransformer([
    ("numericas", StandardScaler(), num_cols),
    ("categoricas", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
X_proc = preprocesador.fit_transform(X)
if hasattr(X_proc, "toarray"):
    X_proc = X_proc.toarray()

# PCA: proyectar todos los registros a 2 dimensiones.
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_proc)

def graficar(X_2d, y, titulo, archivo, varianza=None):
    plt.figure(figsize=(10, 7))
    for clase in sorted(y.unique()):
        mask = y == clase
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=18, alpha=0.65, label=f"Clase {clase}")
    plt.title(titulo)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    if varianza is not None:
        plt.title(f"{titulo}\nVarianza explicada: {varianza[0]:.1%} + {varianza[1]:.1%}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(archivo, dpi=160)
    plt.show()

graficar(X_pca, y, "Historial de precios - PCA", "precios/historial_precios_pca.png",
          pca.explained_variance_ratio_)

# t-SNE se calcula sobre una muestra para reducir tiempo y memoria.
n = min(MAX_TSNE, len(X_proc))
rng = np.random.default_rng(RANDOM_STATE)
indices = rng.choice(len(X_proc), size=n, replace=False) if n < len(X_proc) else np.arange(n)
perplexity = min(30, max(2, (n - 1) // 3))
X_tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
               learning_rate="auto", random_state=RANDOM_STATE).fit_transform(X_proc[indices])
graficar(X_tsne, y.iloc[indices].reset_index(drop=True),
          f"Historial de precios - t-SNE (muestra de {n})", "precios/historial_precios_tsne.png")

print("PCA varianza explicada:", pca.explained_variance_ratio_)
print("Graficos guardados en PNG.")
