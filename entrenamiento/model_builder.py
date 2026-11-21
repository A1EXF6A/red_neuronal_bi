"""Construcción de la red neuronal (misma arquitectura que la versión de un solo script)."""
import sys
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import SEED


def build_model(input_dim, output_dim, kind, cfg):
    tf.keras.utils.set_random_seed(SEED)
    regularizer = tf.keras.regularizers.l2(cfg["l2"])
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for units in cfg["layers"]:
        model.add(tf.keras.layers.Dense(
            units, activation=cfg["activation"], kernel_regularizer=regularizer
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(cfg["dropout"]))
    if kind == "multiclass":
        model.add(tf.keras.layers.Dense(output_dim, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
    else:
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
        loss=loss,
    )
    return model
