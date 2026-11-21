"""Orquestador del entrenamiento de una red: encadena las etapas en el orden correcto."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import TASKS
from data_loader import cargar_datos
from correlaciones import generar_correlaciones
from model_selector import entrenar_configuraciones
from report_best import evaluar_test
from graficacion import guardar_graficos
from exportar_pesos import exportar_pesos
from model_exporter import exportar_todo


def train_task(task_name):
    if task_name not in TASKS:
        raise ValueError(f"Tarea desconocida: {task_name}. Opciones: {list(TASKS)}")

    datos = cargar_datos(task_name)                         # data_loader
    print("\n" + "=" * 78)
    print(f"ENTRENAMIENTO: {task_name.upper()}")
    print(f"Dimensión de entrada tras codificación: {datos['X_train'].shape[1]}")
    print(f"Train={len(datos['y_train']):,} | Validación={len(datos['y_val']):,} | Test={len(datos['y_test']):,}")
    print(f"Clases: {datos['label_encoder'].classes_.tolist()}")

    correlaciones = generar_correlaciones(datos, task_name)  # solo lee train
    seleccion = entrenar_configuraciones(datos, task_name)   # model_selector
    evaluacion = evaluar_test(datos, seleccion, task_name)   # report_best (test, una sola vez)
    graficos = guardar_graficos(datos, seleccion, evaluacion, task_name)  # graficacion
    pesos = exportar_pesos(datos, seleccion, task_name)      # solo el mejor modelo

    return exportar_todo(
        datos, seleccion, evaluacion, task_name,
        extras={
            "correlaciones": correlaciones,
            "pesos_mejor_modelo": pesos,
            "graficos_evaluacion": [str(p.relative_to(ROOT)) for p in graficos],
        },
    )
