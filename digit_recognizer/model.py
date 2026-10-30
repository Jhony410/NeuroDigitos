"""Definición del perceptrón multicapa (MLP) para clasificar dígitos.

Arquitectura:
    Entrada 28x28 -> Flatten (784) -> Dense 128 ReLU -> Dropout 0.2
    -> Dense 64 ReLU -> Dense 10 softmax
"""
from __future__ import annotations

from tensorflow import keras

from . import config


def build_mlp(hidden_units: tuple[int, ...] = config.HIDDEN_UNITS,
              dropout_rate: float = config.DROPOUT_RATE,
              learning_rate: float = config.LEARNING_RATE,
              seed: int = config.SEED) -> keras.Model:
    """Construye y compila el MLP."""
    first, second = hidden_units

    model = keras.Sequential(
        [
            keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE), name="imagen_28x28"),
            # Convierte la matriz 28x28 en un vector de 784 características.
            keras.layers.Flatten(name="aplanado_784"),
            # He normal: inicialización recomendada para ReLU (mantiene la varianza).
            keras.layers.Dense(first, activation="relu",
                               kernel_initializer=keras.initializers.HeNormal(seed=seed),
                               name="oculta_1"),
            # Dropout: apaga al azar el 20 % de neuronas SOLO durante el entrenamiento.
            keras.layers.Dropout(dropout_rate, seed=seed, name="dropout"),
            keras.layers.Dense(second, activation="relu",
                               kernel_initializer=keras.initializers.HeNormal(seed=seed + 1),
                               name="oculta_2"),
            # Softmax: convierte 10 puntajes en probabilidades que suman 1.
            keras.layers.Dense(config.NUM_CLASSES, activation="softmax",
                               kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 2),
                               name="salida_softmax"),
        ],
        name="mlp_digitos",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        # Etiquetas enteras 0..9 (no one-hot) -> sparse categorical crossentropy.
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
