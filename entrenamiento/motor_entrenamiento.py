# Compatibilidad para el entrenamiento
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
