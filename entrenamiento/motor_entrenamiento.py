"""Fachada de compatibilidad: mantiene `from motor_entrenamiento import train_task`.

La lógica real está repartida en: data_loader, correlaciones, model_builder, model_selector,
model_utils, report_best, graficacion, exportar_pesos, model_exporter y experimento.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimento import train_task  # noqa: E402,F401

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tarea", choices=["pedidos", "resenas", "precios"])
    args = parser.parse_args()
    train_task(args.tarea)
