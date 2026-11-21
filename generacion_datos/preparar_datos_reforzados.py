"""
preparar_datos_v6.py  -  GlobalBite Delivery
Genera versiones v6 de los tres datasets, pensadas para redes neuronales:
  - Clases balanceadas (cortes por cuantiles).
  - Variables derivadas nuevas (sin fuga de la etiqueta).
  - Señal sintética reforzada y documentada (ver README_v6.md).
  - Se eliminan columnas con fuga / identificadores / texto.
  - Columna 'split' (train/val/test) estratificada, lista para usar.

Uso:  python preparar_datos_v6.py [carpeta_entrada] [carpeta_salida]
"""
import sys, os, json
import numpy as np
import pandas as pd

IN = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "datos_v6"
os.makedirs(OUT, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)
info = {}


def leer(nombre):
    return pd.read_csv(os.path.join(IN, nombre), encoding="utf-8-sig")


def split_estratificado(y, grupos=None, seed=SEED):
    """70/15/15 estratificado (o por grupos si se dan, para no mezclar un grupo entre splits)."""
    r = np.random.default_rng(seed)
    s = np.empty(len(y), dtype=object)
    if grupos is None:
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            r.shuffle(idx)
            a, b = int(.70 * len(idx)), int(.85 * len(idx))
            s[idx[:a]], s[idx[a:b]], s[idx[b:]] = "train", "val", "test"
    else:
        g = pd.Series(grupos)
        ug = g.unique()
        r.shuffle(ug)
        a, b = int(.70 * len(ug)), int(.85 * len(ug))
        m = {k: "train" for k in ug[:a]}
        m.update({k: "val" for k in ug[a:b]})
        m.update({k: "test" for k in ug[b:]})
        s = g.map(m).values
    return s


def efecto(serie, sd):
    """Efecto aleatorio fijo (reproducible) por categoría, media 0."""
    cats = sorted(serie.dropna().unique())
    e = rng.normal(0, sd, len(cats))
    e -= e.mean()
    return serie.map(dict(zip(cats, e)))


def cortes_cuantiles(x, k):
    qs = np.linspace(0, 1, k + 1)[1:-1]
    return [float(np.quantile(x, q)) for q in qs]


# =====================================================================
# 1. PEDIDOS
# =====================================================================
p = leer("pedidos_etiquetado.csv")
p["fecha"] = pd.to_datetime(p["fecha"])
p["dow"] = p["fecha"].dt.dayofweek
p["tiempo_entrega_orig"] = p["tiempo_entrega_minutos"]

# --- tiempo de entrega: parte sistemática original + factores observables + menos ruido
base = 7.33 + 1.798 * p["distancia_km"] + 19.75 * p["multiplicador_hora_pico"]
rating_ef = p["nivel_rating_restaurante"].map({"Alto": -1.5, "Medio": 0.0, "Bajo": 1.5}).fillna(0.0)
mes_ef = 1.5 * np.cos((p["mes"] - 12) / 12 * 2 * np.pi)            # pico en diciembre
dia_ef = p["dow"].map({4: 1.0, 5: 1.5, 6: 1.0}).fillna(0.0)          # vie/sáb/dom
t = (base
     + efecto(p["ciudad_restaurante"], 2.0)       # congestión por ciudad
     + efecto(p["cocina_restaurante"], 1.2)       # tiempo de preparación por cocina
     + efecto(p["restaurante_id"].astype(str), 1.0)  # efecto propio del restaurante
     + rating_ef + mes_ef + dia_ef
     + rng.normal(0, 2.2, len(p)))                # ruido (antes sd=5.6)
p["tiempo_entrega_minutos"] = np.clip(np.round(t), 12, None).astype(int)

# --- etiqueta balanceada por cuantiles (0 rápida, 1 normal, 2 lenta, 3 muy lenta)
cz = [int(round(c)) for c in cortes_cuantiles(p["tiempo_entrega_minutos"], 4)]
p["etiqueta_nivel_demora"] = np.digitize(p["tiempo_entrega_minutos"], [c + 0.5 for c in cz])
info["pedidos_cortes_minutos"] = cz

# --- calificación operativa coherente con el nuevo tiempo
p["calificacion_operativa"] = np.clip(
    np.round(4.9 - 0.045 * (p["tiempo_entrega_minutos"] - 25) + rng.normal(0, 0.2, len(p)), 2), 2.5, 5.0)

# --- variables derivadas (todas disponibles al momento del pedido)
p["tarifa_por_km"] = p["tarifa_entrega"] / p["distancia_km"]
p["monto_por_km"] = p["monto_total"] / p["distancia_km"]
p["log_distancia"] = np.log1p(p["distancia_km"])
p["es_hora_pico"] = (p["multiplicador_hora_pico"] > 1.0).astype(int)
p["dist_x_pico"] = p["distancia_km"] * p["multiplicador_hora_pico"]
p["cliente_misma_ciudad"] = (p["ciudad_cliente"] == p["ciudad_restaurante"]).astype(int)
p["cliente_mismo_pais"] = (p["pais_cliente"] == p["pais_restaurante"]).astype(int)
p["dow_sin"] = np.sin(2 * np.pi * p["dow"] / 7)
p["dow_cos"] = np.cos(2 * np.pi * p["dow"] / 7)
p["mes_sin"] = np.sin(2 * np.pi * p["mes"] / 12)
p["mes_cos"] = np.cos(2 * np.pi * p["mes"] / 12)
p["antiguedad_cliente_dias"] = (p["fecha"] - pd.to_datetime(p["fecha_registro_cliente"])).dt.days.clip(lower=0)

# historial del restaurante: media de tiempo de SUS pedidos anteriores (sin incluir el actual)
p = p.sort_values(["fecha", "pedido_id"]).reset_index(drop=True)
g = p.groupby("restaurante_id")["tiempo_entrega_minutos"]
cum = g.cumsum() - p["tiempo_entrega_minutos"]
cnt = g.cumcount()
glob = p["tiempo_entrega_minutos"].expanding().mean().shift(1).fillna(p["tiempo_entrega_minutos"].mean())
k = 5  # suavizado hacia la media global
p["hist_tiempo_restaurante"] = (cum + k * glob) / (cnt + k)
p["hist_n_pedidos_restaurante"] = cnt
cg = p.groupby("ciudad_restaurante")["tiempo_entrega_minutos"]
cumc = cg.cumsum() - p["tiempo_entrega_minutos"]
cntc = cg.cumcount()
p["hist_tiempo_ciudad"] = (cumc + k * glob) / (cntc + k)

p["split"] = split_estratificado(p["etiqueta_nivel_demora"].values)

cols_modelo_p = [
    "pedido_id", "split", "etiqueta_nivel_demora",
    "distancia_km", "log_distancia", "tarifa_entrega", "tarifa_por_km", "monto_total", "monto_por_km",
    "multiplicador_hora_pico", "es_hora_pico", "dist_x_pico",
    "es_fin_de_semana", "dow_sin", "dow_cos", "mes_sin", "mes_cos",
    "costo_promedio_dos_personas", "tiene_delivery", "tiene_comer_en_local", "tiene_wifi", "acepta_tarjeta",
    "cliente_misma_ciudad", "cliente_mismo_pais", "antiguedad_cliente_dias",
    "hist_tiempo_restaurante", "hist_n_pedidos_restaurante", "hist_tiempo_ciudad",
    # categóricas (one-hot en el script de entrenamiento)
    "ciudad_restaurante", "cocina_restaurante", "nivel_rating_restaurante",
    "metodo_pago", "segmento_cliente", "pais_restaurante",
]
p.to_csv(os.path.join(OUT, "pedidos_v6_completo.csv"), index=False, encoding="utf-8")
p[cols_modelo_p].to_csv(os.path.join(OUT, "pedidos_v6.csv"), index=False, encoding="utf-8")

# =====================================================================
# 2. RESEÑAS  (hereda del pedido: tiempo, distancia, demora, calificación operativa)
# =====================================================================
r = leer("resenas_etiquetado.csv")
r["fecha"] = pd.to_datetime(r["fecha"])
ped = p[["pedido_id", "tiempo_entrega_minutos", "distancia_km", "multiplicador_hora_pico", "tarifa_entrega",
         "monto_total", "calificacion_operativa", "etiqueta_nivel_demora", "estado_pedido",
         "hist_tiempo_restaurante"]].rename(columns={"etiqueta_nivel_demora": "nivel_demora_pedido"})
r = r.merge(ped, on="pedido_id", how="left", validate="one_to_one")
assert r["tiempo_entrega_minutos"].notna().all()

r["calificacion_orig"] = r["calificacion"]
rating_ef_r = r["nivel_rating_restaurante"].map({"Alto": 0.15, "Medio": 0.0, "Bajo": -0.15}).fillna(0.0)
seg_ef = r["segmento_cliente"].map({"VIP": -0.10}).fillna(0.0)           # clientes VIP más exigentes
cal = (r["calificacion_operativa"] + rating_ef_r + seg_ef
       + efecto(r["cocina_restaurante"], 0.10)
       + rng.normal(0, 0.30, len(r)))
r["calificacion"] = np.clip(np.round(cal, 1), 1.5, 5.0)

cr = [round(c, 1) for c in cortes_cuantiles(r["calificacion"], 4)]
r["etiqueta_nivel_satisfaccion"] = np.digitize(r["calificacion"], [c + 0.05 for c in cr])
info["resenas_cortes_calificacion"] = cr

# comentario coherente con la nueva clase (texto = solo para análisis, NO usar como feature)
pool = {0: ["Poor experience, would not order again.", "Delivery was very delayed.", "Order arrived late and cold.",
            "Not satisfied, wrong items delivered."],
        1: ["Food was okay, delivery took a while.", "It was fine, some items were cold.",
            "Decent but could be better.", "Average experience, nothing special."],
        2: ["Consistently good quality.", "Great food, fast delivery, will order again."],
        3: ["Excellent service and tasty food.", "Loved it, definitely recommend.",
            "Really enjoyed this, everything arrived hot."]}
r["comentario"] = [rng.choice(pool[c]) if tc == 1 else np.nan
                   for c, tc in zip(r["etiqueta_nivel_satisfaccion"], r["tiene_comentario"])]

r["dow"] = r["fecha"].dt.dayofweek
r["dow_sin"] = np.sin(2 * np.pi * r["dow"] / 7)
r["dow_cos"] = np.cos(2 * np.pi * r["dow"] / 7)
r["mes_sin"] = np.sin(2 * np.pi * r["mes"] / 12)
r["mes_cos"] = np.cos(2 * np.pi * r["mes"] / 12)
r["tarifa_por_km"] = r["tarifa_entrega"] / r["distancia_km"]
r["tiempo_vs_esperado_restaurante"] = r["tiempo_entrega_minutos"] - r["hist_tiempo_restaurante"]
r["es_hora_pico"] = (r["multiplicador_hora_pico"] > 1.0).astype(int)

r["split"] = split_estratificado(r["etiqueta_nivel_satisfaccion"].values)
cols_modelo_r = [
    "resena_key", "pedido_id", "split", "etiqueta_nivel_satisfaccion",
    "tiempo_entrega_minutos", "distancia_km", "multiplicador_hora_pico", "es_hora_pico", "tarifa_entrega",
    "tarifa_por_km", "monto_total", "calificacion_operativa", "nivel_demora_pedido",
    "tiempo_vs_esperado_restaurante", "hist_tiempo_restaurante",
    "es_fin_de_semana", "dow_sin", "dow_cos", "mes_sin", "mes_cos", "tiene_comentario",
    "ciudad_restaurante", "cocina_restaurante", "nivel_rating_restaurante", "segmento_cliente", "pais_restaurante",
]
r.to_csv(os.path.join(OUT, "resenas_v6_completo.csv"), index=False, encoding="utf-8")
r[cols_modelo_r].to_csv(os.path.join(OUT, "resenas_v6.csv"), index=False, encoding="utf-8")

# =====================================================================
# 3. PRECIOS
# =====================================================================
h = leer("historial_precios_etiquetado.csv")
h["fecha"] = pd.to_datetime(h["fecha"])
h = h.sort_values(["item_menu_id", "fecha"]).reset_index(drop=True)
h["pct_orig"] = h["porcentaje_cambio"]
h["n_cambio"] = h.groupby("item_menu_id").cumcount()          # 0 = primer cambio, 1 = segundo
h["dias_desde_inicio"] = (h["fecha"] - h["fecha"].min()).dt.days
h["dias_desde_anterior"] = h.groupby("item_menu_id")["fecha"].diff().dt.days

cat_ef = efecto(h["categoria_item"], 1.2)
city_ef = efecto(h["ciudad_restaurante"], 1.5)             # inflación local por ciudad
cuis_ef = efecto(h["cocina_item"], 0.8)
season = 0.5 * np.cos((h["mes"] - 12) / 12 * 2 * np.pi)
lp = np.log(h["precio_anterior"])
precio_ef = -2.0 * (lp - lp.mean()) / lp.std()             # los items baratos suben más en %
noise = rng.normal(0, 2.0, len(h))

# generación secuencial por ítem (el 2º cambio depende del 1º y parte de su precio nuevo)
pa = h["precio_anterior"].values.copy()
pct = np.zeros(len(h))
lag = np.full(len(h), np.nan)
idx0 = np.where(h["n_cambio"].values == 0)[0]
idx1 = np.where(h["n_cambio"].values == 1)[0]
mu = (9.0 + cat_ef + city_ef + cuis_ef + season + precio_ef).values
pct[idx0] = mu[idx0] + noise[idx0]
pct[idx0] = np.clip(pct[idx0], -2, 27)
nuevo0 = np.round(pa[idx0] * (1 + pct[idx0] / 100), 2)
pa[idx1] = nuevo0                                           # encadenar precios
lag[idx1] = pct[idx0]
pct[idx1] = np.clip(mu[idx1] - 0.45 * (pct[idx0] - 10) + noise[idx1], -2, 27)
nuevo = np.round(pa * (1 + pct / 100), 2)

h["precio_anterior"] = pa
h["precio_nuevo"] = nuevo
h["monto_cambio"] = np.round(nuevo - pa, 2)
h["porcentaje_cambio"] = np.round((nuevo / pa - 1) * 100, 2)
h["lag_pct"] = lag
h["lag_pct"] = h["lag_pct"].fillna(h["lag_pct"].mean())
h["tiene_historial"] = (h["n_cambio"] > 0).astype(int)

umbral = round(float(h["porcentaje_cambio"].median()), 1)
h["etiqueta_cambio_fuerte"] = (h["porcentaje_cambio"] > umbral).astype(int)
info["precios_umbral_pct"] = umbral

h["log_precio_anterior"] = np.log(h["precio_anterior"])
h["mes_sin"] = np.sin(2 * np.pi * h["mes"] / 12)
h["mes_cos"] = np.cos(2 * np.pi * h["mes"] / 12)
h["dias_desde_anterior"] = h["dias_desde_anterior"].fillna(h["dias_desde_anterior"].median())
# inflación reciente de la ciudad: media de % de cambios PREVIOS en esa ciudad (sin el actual)
h = h.sort_values(["fecha", "cambio_precio_key"]).reset_index(drop=True)
gc = h.groupby("ciudad_restaurante")["porcentaje_cambio"]
cs = gc.cumsum() - h["porcentaje_cambio"]
cn = gc.cumcount()
gm = h["porcentaje_cambio"].expanding().mean().shift(1).fillna(h["porcentaje_cambio"].mean())
h["hist_pct_ciudad"] = (cs + 20 * gm) / (cn + 20)
gk = h.groupby("categoria_item")["porcentaje_cambio"]
ks = gk.cumsum() - h["porcentaje_cambio"]
kn = gk.cumcount()
h["hist_pct_categoria"] = (ks + 20 * gm) / (kn + 20)

h["split"] = split_estratificado(h["etiqueta_cambio_fuerte"].values, grupos=h["item_menu_id"].values)
cols_modelo_h = [
    "cambio_precio_key", "split", "etiqueta_cambio_fuerte",
    "precio_anterior", "log_precio_anterior", "n_cambio", "tiene_historial", "lag_pct", "dias_desde_anterior",
    "dias_desde_inicio", "mes_sin", "mes_cos", "anio", "disponible_item",
    "hist_pct_ciudad", "hist_pct_categoria",
    "categoria_item", "cocina_item", "ciudad_restaurante", "pais_restaurante",
]
h.to_csv(os.path.join(OUT, "historial_precios_v6_completo.csv"), index=False, encoding="utf-8")
h[cols_modelo_h].to_csv(os.path.join(OUT, "historial_precios_v6.csv"), index=False, encoding="utf-8")

# =====================================================================
# Resumen
# =====================================================================
for nombre, df, col in [("pedidos", p, "etiqueta_nivel_demora"),
                        ("resenas", r, "etiqueta_nivel_satisfaccion"),
                        ("precios", h, "etiqueta_cambio_fuerte")]:
    vc = df[col].value_counts(normalize=True).sort_index().round(4)
    info[f"{nombre}_distribucion"] = {int(k): float(v) for k, v in vc.items()}
    info[f"{nombre}_filas"] = int(len(df))
with open(os.path.join(OUT, "resumen_v6.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(json.dumps(info, indent=2, ensure_ascii=False))

# =====================================================================
# 4. CSV finales "*_reforzado_etiquetado.csv" (todas las columnas, etiqueta al final)
# =====================================================================
_cfg = {"pedidos": (["hubo_demora"], "etiqueta_nivel_demora"),
        "resenas": ([], "etiqueta_nivel_satisfaccion"),
        "historial_precios": ([], "etiqueta_cambio_fuerte")}
for _f, (_extra, _y) in _cfg.items():
    _d = pd.read_csv(os.path.join(OUT, f"{_f}_v6_completo.csv"))
    _d = _d.drop(columns=["split", "dow", "etiqueta_hubo_demora"], errors="ignore")
    _fin = _extra + [_y]
    _d = _d[[c for c in _d.columns if c not in _fin] + _fin]
    _d.to_csv(os.path.join(OUT, f"{_f}_reforzado_etiquetado.csv"), index=False, encoding="utf-8")
print("CSV reforzados generados en", OUT)
