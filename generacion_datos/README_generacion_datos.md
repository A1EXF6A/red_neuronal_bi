# GlobalBite – Datos v6 para redes neuronales

## Diagnóstico (por qué la v5 rendía mal)
Un modelo de árboles (HistGradientBoosting) con todas las variables válidas daba el mismo resultado que tu red
(pedidos ≈ 0.67 acc / 0.62 F1). La red no era el problema: **los datos no contenían suficiente señal**.

- **Pedidos:** la etiqueta es solo `tiempo_entrega_minutos` cortado en 45/60/70 min. Ese tiempo es ~ `7.3 + 1.8·km + 19.8·multiplicador`
  más un ruido aleatorio de ±5.6 min (R² = 0.79). Ciudad, cocina, día, rating, etc. casi no influían (< 1 min).
  Con cortes de ~10 min y ruido de 5.6 min, el techo era ~0.67.
- **Reseñas:** la etiqueta son cortes de `calificacion`. Tu red solo veía columnas de la reseña; lo que explica la nota
  (tiempo de entrega, distancia, demora, calificación operativa) estaba en `pedidos` y no se había unido (se une por `pedido_id`).
  `comentario` es texto generado a partir de la clase (fuga): no usar como feature.
- **Precios:** la etiqueta es `porcentaje_cambio > 10`, y ese % es casi aleatorio respecto a las variables disponibles.
  Cada ítem tiene exactamente 2 cambios encadenados; el historial (cambio previo) sí aporta señal y no se usaba.

## Qué hace `preparar_datos_v6.py`
| Dataset | Balance de clases | Cambios principales |
|---|---|---|
| pedidos | 25.1 / 25.0 / 25.2 / 24.7 % (cortes 42, 51, 60 min por cuantiles) | Tiempo regenerado con efectos de ciudad, cocina, rating, mes, día, restaurante y ruido 2.2 min (antes 5.6). Features: `tarifa_por_km`, `es_hora_pico`, `dist_x_pico`, `cliente_misma_ciudad`, `hist_tiempo_restaurante`, `hist_tiempo_ciudad` (solo pedidos previos, sin fuga), seno/coseno de día y mes. |
| reseñas | 27.4 / 27.7 / 20.9 / 24.0 % (cortes 3.2, 3.7, 4.2) | Unión con features del pedido. Calificación regenerada en función del tiempo de entrega (corr = −0.81). Comentario coherente con la nueva clase. |
| precios | 50.4 / 49.6 % (umbral = mediana, 9.1 %) | % de cambio con efectos de categoría, ciudad, estacionalidad, nivel de precio y reversión al cambio previo. Features: `n_cambio`, `lag_pct`, `dias_desde_anterior`, `hist_pct_ciudad`, `hist_pct_categoria`. Precios re-encadenados. |

Los CSV finales `*_reforzado_etiquetado.csv` no incluyen columna `split`: la partición cronológica la hace el proyecto.

**Archivos `*_v6.csv`** = tabla lista para modelar (sin IDs de fuga, sin `tiempo_entrega_minutos`, `calificacion_operativa` ni `calificacion` en pedidos).
**Archivos `*_v6_completo.csv`** = todas las columnas, para análisis de negocio.
Las categóricas (ciudad, cocina, etc.) van como texto: aplícales one-hot en tu script.
Escala numéricas con StandardScaler ajustado solo en `train`.

## Resultados (MLP de scikit-learn sobre la partición cronológica 70/15/15 del proyecto, test)
| Dataset | v5 macro-F1 | reforzado macro-F1 |
|---|---|---|
| Pedidos (4 clases) | 0.624 | **≈ 0.83** |
| Reseñas (4 clases) | 0.380 | **≈ 0.52** (≈ 0.60 si se incluyen `calificacion_operativa` y `nivel_rating_restaurante`) |
| Precios (2 clases) | 0.627 | **≈ 0.69** |

Medido con scikit-learn, no con Keras; con la red Keras del proyecto los valores pueden variar algunos puntos.
Las etiquetas cambiaron (cortes por cuantiles): el baseline mayoritario ahora es ≈ 25 % (4 clases) y ≈ 50 % (2 clases).
En precios se quitó el sesgo temporal (el primer cambio de cada ítem subía más que el segundo) para que
las clases queden balanceadas también en la partición cronológica (test 56/44 %).

## Advertencias
- La mejora viene en buena parte de **reforzar la señal sintética**. Debe declararse en el informe como datos sintéticos
  regenerados (columnas originales conservadas como `tiempo_entrega_orig`, `pct_orig`, `calificacion_orig` en los `_completo`).
- Cortes por cuantiles = niveles *relativos* (rápida/normal/lenta/muy lenta), no umbrales de negocio fijos.
- En los datos originales la **cancelación no depende del tiempo de entrega** (pregunta de negocio 1); v6 no lo corrige.
- No uses `comentario` como entrada del modelo de reseñas (es proxy de la etiqueta).
