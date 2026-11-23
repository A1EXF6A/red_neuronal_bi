# Ejecuta el flujo completo
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

COMMANDS = [
    [PYTHON, str(ROOT / "preprocesamiento" / "procesar_pedidos.py")],
    [PYTHON, str(ROOT / "preprocesamiento" / "procesar_resenas.py")],
    [PYTHON, str(ROOT / "preprocesamiento" / "procesar_precios.py")],
    [PYTHON, str(ROOT / "entrenamiento" / "entrenar_pedidos.py")],
    [PYTHON, str(ROOT / "entrenamiento" / "entrenar_resenas.py")],
    [PYTHON, str(ROOT / "entrenamiento" / "entrenar_precios.py")],
]

# Ejecuta todas las tareas
def main():
    for number, command in enumerate(COMMANDS, start=1):
        print("\\n" + "=" * 80, flush=True)
        print(f"ETAPA {number}/{len(COMMANDS)}: {' '.join(command[1:])}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\\nPROCESO COMPLETO: preparación y entrenamiento de las tres redes finalizados.")

if __name__ == "__main__":
    main()
