"""Inferencia compartida por la interfaz y la consola, sin entrenar el modelo."""
from pathlib import Path

import numpy as np

from . import config


def load_predictor(path: Path = config.MODEL_PATH):
    from tensorflow import keras

    if not Path(path).is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {path}. Ejecute python train.py")
    model = keras.models.load_model(path, compile=False)
    if model.input_shape != (None, 28, 28) or model.output_shape != (None, 10):
        raise ValueError("Modelo incompatible: se requiere entrada (28, 28) y 10 salidas.")
    return model


def predict_probabilities(model, batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(batch, dtype=np.float32)
    if batch.ndim != 3 or batch.shape[1:] != (28, 28) or len(batch) == 0:
        raise ValueError("La entrada debe tener forma (N, 28, 28), con N mayor que cero.")
    if not np.isfinite(batch).all() or batch.min() < 0 or batch.max() > 1:
        raise ValueError("La entrada debe contener valores finitos entre 0 y 1.")
    probs = np.asarray(model(batch, training=False))
    if (probs.shape != (len(batch), 10) or not np.isfinite(probs).all()
            or (probs < 0).any() or (probs > 1).any()
            or not np.allclose(probs.sum(axis=1), 1, atol=1e-4)):
        raise ValueError("El modelo no devuelve diez probabilidades válidas por imagen.")
    return probs
