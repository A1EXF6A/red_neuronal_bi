# Carga de datos y preprocesador
import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from configuracion import DATA_PROCESSED, TASKS


# Construye el preprocesador
def make_preprocessor(X_train):
    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


# Carga los datos preparados
def cargar_datos(task_name):
    # Devuelve los datos preparados para el entrenamiento
    task = TASKS[task_name]
    split_path = DATA_PROCESSED / task_name / "particiones.joblib"
    if not split_path.exists():
        raise FileNotFoundError(
            f"No existen datos preparados para {task_name}. "
            f"Ejecuta primero preprocesamiento/procesar_{task_name}.py"
        )
    bundle = joblib.load(split_path)

    X_train_df = bundle["X_train"]
    X_val_df = bundle["X_validation"]
    X_test_df = bundle["X_test"]
    y_train_raw = bundle["y_train"].astype(str)
    y_val_raw = bundle["y_validation"].astype(str)
    y_test_raw = bundle["y_test"].astype(str)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    known_classes = set(label_encoder.classes_)
    unseen = (set(y_val_raw.unique()) | set(y_test_raw.unique())) - known_classes
    if unseen:
        raise ValueError(f"Clases no vistas durante entrenamiento en {task_name}: {sorted(unseen)}")
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    # Ajuste con train
    preprocessor = make_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df).astype("float32")
    X_val = preprocessor.transform(X_val_df).astype("float32")
    X_test = preprocessor.transform(X_test_df).astype("float32")

    return {
        "task": task,
        "bundle": bundle,
        "X_train_df": X_train_df,
        "X_val_df": X_val_df,
        "X_test_df": X_test_df,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "preprocessor": preprocessor,
        "feature_names": [str(n) for n in preprocessor.get_feature_names_out()],
    }
