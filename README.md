# Proyecto modular V3 — tres redes neuronales

Esta versión reorganiza el proyecto V3 en dos etapas independientes: preparación de datos y entrenamiento. Incluye un ejecutor general para automatizar ambas etapas.

## Estructura

```text
proyecto/
├── datos/
│   ├── originales/
│   │   ├── pedidos_reforzado_etiquetado.csv
│   │   ├── resenas_reforzado_etiquetado.csv
│   │   └── historial_precios_reforzado_etiquetado.csv
│   └── procesados/
│       ├── pedidos/    (particiones.joblib, resumen_preparacion.json, datos_procesados.csv)
│       ├── resenas/
│       └── precios/
├── preprocesamiento/
│   ├── motor_preprocesamiento.py
│   ├── procesar_pedidos.py
│   ├── procesar_resenas.py
│   └── procesar_precios.py
├── entrenamiento/
│   ├── entrenar_pedidos.py        (punto de entrada de cada red)
│   ├── entrenar_resenas.py
│   ├── entrenar_precios.py
│   ├── motor_entrenamiento.py     (fachada: reexporta train_task)
│   ├── experimento.py             (orquestador de las etapas)
│   ├── data_loader.py             (carga, etiquetas y preprocesador solo con train)
│   ├── correlaciones.py           (matrices de correlación, solo train)
│   ├── model_builder.py           (arquitectura de la red)
│   ├── model_selector.py          (bucle de configuraciones, mejor por Macro-F1)
│   ├── model_utils.py             (métricas, pesos de clase, predicción)
│   ├── report_best.py             (evaluación única en test, CSV/JSON)
│   ├── graficacion.py             (todos los gráficos)
│   ├── exportar_pesos.py          (pesos de capas ocultas del mejor modelo)
│   └── model_exporter.py          (.keras, .joblib y metadata)
├── modelos/
│   ├── modelo_pedidos.keras
│   ├── modelo_resenas.keras
│   ├── modelo_precios.keras
│   ├── artefactos/
│   └── pesos/
│       ├── pedidos/
│       ├── resenas/
│       └── precios/
├── resultados/
│   ├── pedidos/
│   │   └── correlaciones/
│   ├── resenas/
│   │   └── correlaciones/
│   └── precios/
│       └── correlaciones/
├── graficos/
│   ├── pedidos/
│   ├── resenas/
│   └── precios/
├── generacion_datos/        (script que genera los CSV reforzados, ver abajo)
├── configuracion.py
├── ejecutar_todo.py
└── requirements.txt
```

## Cambios de esta versión (CSV reforzados)

El proyecto ahora lee `*_reforzado_etiquetado.csv` (carpeta `datos/originales/`) en lugar de los CSV antiguos.
Estos archivos traen clases balanceadas, variables derivadas nuevas y la etiqueta como última columna.
Cómo se generaron y por qué: `generacion_datos/README_generacion_datos.md`.

Cambios de código:
- `configuracion.py`: nuevos nombres de archivo y `COLUMNAS_PROHIBIDAS` (lista por tarea de columnas con fuga que nunca entran al modelo).
- `preprocesamiento/motor_preprocesamiento.py`:
  - Pedidos: usa las variables derivadas (`tarifa_por_km`, `dist_x_pico`, `cliente_misma_ciudad`, `hist_tiempo_restaurante`, `hist_tiempo_ciudad`, etc.).
  - Reseñas: ya no cruza con el CSV de pedidos; el CSV reforzado trae unidas las variables del pedido.
  - Precios: se descarta `pct_orig` (porcentaje del dataset original) y se conservan los históricos (`lag_pct`, `n_cambio`, `hist_pct_*`).
- Sin cambios: partición cronológica 70/15/15, ajuste de imputador/escalador/OneHot solo con train, configuraciones de red y métricas.

Etiquetas (cortes por cuantiles, clases balanceadas):
- Pedidos `etiqueta_nivel_demora`: 0 ≤ 42 min, 1 ≤ 51, 2 ≤ 60, 3 > 60.
- Reseñas `etiqueta_nivel_satisfaccion`: 0 ≤ 3.2, 1 ≤ 3.7, 2 ≤ 4.2, 3 > 4.2 de calificación.
- Precios `etiqueta_cambio_fuerte`: 1 si el cambio de precio supera 9.1 %.

Columnas excluidas a propósito: `tiempo_entrega_minutos`, `calificacion_operativa`, `hubo_demora` y `estado_*` en pedidos;
`calificacion`, `comentario`, `calificacion_operativa` y `nivel_rating_restaurante` en reseñas (para incluir
`calificacion_operativa` y `nivel_rating_restaurante` basta quitarlas de `COLUMNAS_PROHIBIDAS["resenas"]`);
`precio_nuevo`, `monto_cambio`, `porcentaje_cambio` y `pct_orig` en precios.

Nota: estos CSV contienen señal sintética reforzada; debe declararse en el informe como datos sintéticos regenerados.

## Instalación

Se recomienda Python 3.10 o 3.11.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Ejecutar todo automáticamente

Desde la raíz del proyecto:

```bash
python ejecutar_todo.py
```

El ejecutor realiza, en orden:
1. Preparación cronológica de pedidos.
2. Preparación cronológica de reseñas.
3. Preparación cronológica de precios.
4. Entrenamiento de pedidos.
5. Entrenamiento de reseñas.
6. Entrenamiento de precios.

Si una etapa falla, el proceso se detiene para no entrenar con datos incompletos.

## Ejecutar etapas por separado

Preparar un dataset:
```bash
python preprocesamiento/procesar_pedidos.py
python preprocesamiento/procesar_resenas.py
python preprocesamiento/procesar_precios.py
```

Entrenar una red (requiere que sus datos ya estén preparados):
```bash
python entrenamiento/entrenar_pedidos.py
python entrenamiento/entrenar_resenas.py
python entrenamiento/entrenar_precios.py
```

También se puede preparar los tres datasets mediante:
```bash
python preprocesamiento/motor_preprocesamiento.py todos
```

## Entrenamiento modular, correlaciones y pesos

El antiguo `motor_entrenamiento.py` (un solo script) se dividió siguiendo la organización de `Proyecto_sexto`
(`data_loader`, `model_builder`, `model_selector`, `model_utils`, `experimento`, `report_best`), pero con módulos
compartidos por las tres redes en lugar de copiarlos en cada carpeta. La metodología, las configuraciones, las
métricas y los nombres de los archivos de salida existentes no cambian.

Flujo de `experimento.train_task()`:

1. `data_loader.cargar_datos`: lee `particiones.joblib`, codifica etiquetas y ajusta el preprocesador solo con train.
2. `correlaciones.generar_correlaciones`: matrices de correlación (solo train).
3. `model_selector.entrenar_configuraciones`: entrena las configuraciones de `configuracion.CONFIGS`.
4. `report_best.evaluar_test`: evalúa en test una sola vez, tras elegir la mejor configuración.
5. `graficacion.guardar_graficos`: gráficos de evaluación.
6. `exportar_pesos.exportar_pesos`: pesos del mejor modelo.
7. `model_exporter.exportar_todo`: `.keras`, `.joblib` y `metadata.json`.

### Gráficos por red: `graficos/<tarea>/`

| Archivo | Contenido |
|---|---|
| `matriz_confusion_test.png` | Matriz de confusión en test. |
| `comparacion_experimentos.png` | Macro-F1 de validación por configuración. |
| `correlacion_variables.png` | Heatmap de correlación (Pearson) entre variables numéricas originales. |
| `correlacion_con_etiqueta.png` | Top 20 de variables más correlacionadas (Spearman) con la etiqueta. |
| `pesos_capa_oculta_XX.png` | Heatmap de la matriz W de cada capa oculta del mejor modelo. |

### Resultados por red: `resultados/<tarea>/`

| Archivo | Contenido |
|---|---|
| `experimentos.csv` | Tabla de las configuraciones probadas, ordenada por Macro-F1 de validación. |
| `matriz_confusion_test.csv` | Matriz de confusión en test. |
| `reporte_test.json` | Reporte de clasificación en test. |
| `metadata.json` | Metadatos del modelo seleccionado, métricas y rutas de lo generado. |
| `correlaciones/correlacion_numericas.csv` | Pearson entre variables numéricas originales (antes de One-Hot). |
| `correlaciones/correlacion_codificadas.csv` | Pearson entre todas las columnas que recibe la red (tras One-Hot). |
| `correlaciones/correlacion_con_etiqueta.csv` | Spearman de cada columna codificada con la etiqueta, ordenado por magnitud. |

Todas se calculan solo con el bloque train. Para etiquetas multiclase se usa la etiqueta como nivel ordinal (0, 1, 2, 3).

### Pesos del mejor modelo: `modelos/pesos/<tarea>/`

Solo se exportan los pesos del modelo seleccionado (con los de su mejor época), no los de las demás configuraciones.
La cantidad de capas ocultas depende del modelo ganador, por lo que los archivos `capa_oculta_XX` se generan dinámicamente
y se borran los de ejecuciones anteriores.

- `capa_oculta_XX_W.csv` y `capa_oculta_XX_b.csv`: matriz de pesos (filas = entradas, columnas = neuronas) y sesgos.
  En la capa 1 las filas llevan el nombre de la variable de entrada.
- `capa_oculta_XX_batchnorm.csv`: gamma, beta, media móvil y varianza móvil de la BatchNormalization de ese bloque.
- `capa_salida_W.csv` y `capa_salida_b.csv`: capa de salida (no oculta), por separado.
- `pesos_<tarea>.npz`: todos los arrays en un solo archivo para cargar con NumPy.
- `pesos_<tarea>.json`: mismo contenido en JSON (formato parecido al de `Proyecto_sexto`), con arquitectura, columnas de entrada y clases.

Cada bloque oculto es `Dense -> BatchNorm -> Dropout`. Para reproducir una predicción a mano hay que aplicar los parámetros
de BatchNorm después de cada Dense (Dropout solo actúa durante el entrenamiento).

Los `metadata.json` incluyen tres campos nuevos (`correlaciones`, `pesos_mejor_modelo`, `graficos_evaluacion`) con las rutas generadas.

## Metodología conservada de V3 (sin cambios)

- Pedidos: clasificación multiclase con `etiqueta_nivel_demora`, predicción al recibir el pedido.
- Reseñas: clasificación multiclase con `etiqueta_nivel_satisfaccion`, predicción después de la entrega y antes de conocer calificación/comentario.
- Precios: clasificación binaria con `etiqueta_cambio_fuerte`, predicción antes de aplicar el nuevo precio.
- La etiqueta binaria de pedidos no se usa.
- División cronológica 70% train, 15% validación y 15% test.
- El bloque de test contiene las fechas más recientes y solo se utiliza para evaluación final.
- Ingeniería determinista de variables de fecha y variables históricas antes de la división. Las variables históricas de precios usan desplazamiento temporal para excluir el evento actual.
- Imputación, StandardScaler y OneHotEncoder se ajustan en el script de entrenamiento únicamente con train y luego se aplican sin reajuste a validación y test.
- LabelEncoder se ajusta con las etiquetas del bloque train.
- Pesos de clase balanceados calculados solo con train.
- Selección de configuración por Macro-F1 de validación.
- EarlyStopping, ReduceLROnPlateau, BatchNormalization, Dropout y regularización L2.
- Salida softmax para multiclase y sigmoide para binaria.

## ¿Qué se guarda?

### datos/procesados/
Cada subcarpeta (`pedidos/`, `resenas/`, `precios/`) contiene:
- `particiones.joblib`: características y etiquetas separadas cronológicamente en train, validation y test.
- `resumen_preparacion.json`: conteos, distribución de clases y columnas (más `archivo_csv`, `columna_etiqueta_csv` y `columna_particion_csv`).
- `datos_procesados.csv`: un único CSV con todos los registros para inspeccionarlos o usarlos fuera de Python.
  Columnas: `fecha` (solo de referencia, no es una característica), `particion` (`train`, `validation` o `test`),
  las variables preparadas y la etiqueta como última columna.

Estos datos (joblib y CSV) son características ya preparadas mediante ingeniería determinista y limpieza. No contienen aún imputación, escalado ni One-Hot aprendidos; esos pasos se ajustan durante el entrenamiento exclusivamente con train para evitar fuga de información. El CSV se escribe sin esas transformaciones y sus filas van en orden cronológico (train = fechas más antiguas, test = más recientes).

### modelos/
- Tres modelos `.keras`.
- En `modelos/artefactos/`: preprocesadores `.joblib` y codificadores de etiquetas `.joblib`.
- En `modelos/pesos/<tarea>/`: pesos de las capas del mejor modelo (CSV, NPZ y JSON).

### resultados/
Una subcarpeta por red (`pedidos/`, `resenas/`, `precios/`) con la tabla de experimentos, matriz de confusión CSV,
reporte de clasificación JSON, metadatos del modelo y `correlaciones/`. Ver el detalle en la sección anterior.

### graficos/
Una subcarpeta por red (`pedidos/`, `resenas/`, `precios/`) con matriz de confusión, comparación de experimentos,
correlaciones y heatmaps de pesos. Ver el detalle en la sección anterior.

## Importante para predecir nuevos registros

Para inferencia se deben aplicar las mismas transformaciones de ingeniería de variables que se usaron al preparar el dataset, y luego cargar el preprocesador y codificador `.joblib` correspondientes. El modelo `.keras` por sí solo no incluye esas transformaciones.
