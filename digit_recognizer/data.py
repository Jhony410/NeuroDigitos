"""Carga y preparación del dataset MNIST.

- Conserva intacto el conjunto de prueba oficial (10 000 imágenes).
- Obtiene la validación SOLO del conjunto de entrenamiento, con
  división estratificada para mantener la proporción de cada dígito.
- Normaliza los píxeles de [0, 255] a [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from . import config


@dataclass
class MnistSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_raw_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Devuelve MNIST en uint8 con la misma forma que keras.datasets.mnist.

    Si existe data/mnist.npz se usa esa copia local (útil sin internet);
    en caso contrario Keras descarga el archivo oficial y lo guarda en caché.
    """
    if config.LOCAL_MNIST_PATH.exists():
        with np.load(config.LOCAL_MNIST_PATH, allow_pickle=False) as data:
            for partition, count in (("train", 60000), ("test", 10000)):
                x, y = data[f"x_{partition}"], data[f"y_{partition}"]
                if (x.shape != (count, 28, 28) or y.shape != (count,)
                        or x.dtype != np.uint8 or not np.issubdtype(y.dtype, np.integer)
                        or y.min() < 0 or y.max() >= config.NUM_CLASSES):
                    raise ValueError(f"MNIST local inválido: revise la partición {partition}.")
            return (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])

    from tensorflow import keras  # importación diferida: es lenta

    return keras.datasets.mnist.load_data()


def normalize(images: np.ndarray) -> np.ndarray:
    """Escala píxeles uint8 [0, 255] a float32 [0, 1]."""
    return images.astype("float32") / 255.0


def load_splits(validation_fraction: float = config.VALIDATION_FRACTION,
                seed: int = config.SEED) -> MnistSplits:
    """Carga MNIST y crea las particiones entrenamiento / validación / prueba."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction debe estar entre 0 y 1, sin incluir extremos.")
    (x_full, y_full), (x_test, y_test) = load_raw_mnist()

    x_train, x_val, y_train, y_val = train_test_split(
        x_full,
        y_full,
        test_size=validation_fraction,
        stratify=y_full,          # misma distribución de clases en ambos lados
        random_state=seed,
    )

    return MnistSplits(
        x_train=normalize(x_train), y_train=y_train.astype("int64"),
        x_val=normalize(x_val), y_val=y_val.astype("int64"),
        x_test=normalize(x_test), y_test=y_test.astype("int64"),
    )
