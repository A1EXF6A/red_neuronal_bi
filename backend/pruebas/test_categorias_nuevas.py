"""Valores nuevos (que no estaban en el entrenamiento) deben poder predecirse.

Verifica validador + ingeniería de variables + preprocesador .joblib (la entrada real de la red).
No necesita TensorFlow ni FastAPI.   Uso:  python -m backend.pruebas.test_categorias_nuevas
"""
from __future__ import annotations

import numpy as np

from backend.servicios.cargador_modelos import cargar_artefactos
from backend.servicios.ingenieria_variables import construir_fila
from backend.servicios.recursos import cargar_esquema, referencias
from backend.servicios.validador import avisos_categorias_nuevas, validar

BASE = {
    "pedidos": {
        "fecha": "2026-08-20", "monto_total": 84.6, "tarifa_entrega": 9.5, "distancia_km": 10.9,
        "multiplicador_hora_pico": 1.25, "metodo_pago": "Wallet", "ciudad_cliente": "Chicago",
        "segmento_cliente": "Recurrente", "metodo_pago_preferido_cliente": "Wallet",
        "fecha_registro_cliente": "2025-01-10", "cliente_activo": True, "ciudad_restaurante": "Chicago",
        "cocina_restaurante": "Italian", "nivel_rating_restaurante": "Alto",
        "costo_promedio_dos_personas": 40, "tiene_delivery": True, "tiene_comer_en_local": True,
        "tiene_wifi": True, "acepta_tarjeta": True, "restaurante_activo": True,
    },
    "resenas": {
        "fecha": "2026-08-20", "tiempo_entrega_minutos": 45, "distancia_km": 8, "multiplicador_hora_pico": 1.0,
        "tarifa_entrega": 6, "monto_total": 50, "ciudad_cliente": "Chicago", "segmento_cliente": "Recurrente",
        "ciudad_restaurante": "Chicago", "cocina_restaurante": "Italian",
    },
    "precios": {
        "fecha": "2026-08-20", "precio_anterior": 12.5, "categoria_item": "Plato principal",
        "cocina_item": "Italian", "disponible_item": True, "cocina_restaurante": "Italian",
        "ciudad_restaurante": "Chicago", "tiene_historial": False,
    },
}
# Valores que NO existen en el entrenamiento (incluye ciudad nueva con y sin país).
NUEVOS = {
    "pedidos": {"ciudad_cliente": "Quito", "pais_cliente": "Ecuador", "ciudad_restaurante": "Ambato",
                "cocina_restaurante": "Ecuadorian", "segmento_cliente": "VIP Galáctico", "metodo_pago": "Cripto"},
    "resenas": {"ciudad_cliente": "Quito", "ciudad_restaurante": "Ambato", "cocina_restaurante": "Ecuadorian",
                "segmento_cliente": "Nuevo-Plus"},
    "precios": {"ciudad_restaurante": "Píllaro", "pais_restaurante": "Ecuador", "cocina_item": "Andean",
                "cocina_restaurante": "Andean", "categoria_item": "Postre raro"},
}


def probar(tarea: str) -> None:
    art, esquema, ref = cargar_artefactos(tarea), cargar_esquema(tarea), referencias()
    datos = {**BASE[tarea], **NUEVOS[tarea]}
    valores, errores = validar(esquema, datos, art.indice_categorias)
    assert not errores, f"{tarea}: se rechazaron valores nuevos: {errores}"
    avisos = avisos_categorias_nuevas(esquema, valores, art.indice_categorias)
    fila = construir_fila(tarea, valores, ref, art.medianas)
    X = art.preprocesador.transform(fila.df[art.columnas]).astype("float32")
    assert X.shape[1] == int(art.metadata["dimensiones_entrada"])
    assert np.isfinite(X).all(), f"{tarea}: la entrada de la red contiene NaN/inf"
    print(f"OK {tarea}: {len(avisos)} avisos de categorías nuevas, entrada {X.shape}")

    # Un valor conocido con otra capitalización/acentos NO debe generar aviso.
    valores2, _ = validar(esquema, {**BASE[tarea], "ciudad_restaurante": " chicago "}, art.indice_categorias)
    assert valores2["ciudad_restaurante"] == "Chicago"


if __name__ == "__main__":
    for t in ("pedidos", "resenas", "precios"):
        probar(t)
    print("Todas las pruebas de categorías nuevas pasaron.")
