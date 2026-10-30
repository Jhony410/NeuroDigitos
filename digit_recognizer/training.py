"""Entrenamiento del MLP y guardado del modelo, historial y resumen."""
from __future__ import annotations

import json
import platform
import time

import numpy as np

from . import config
from .data import load_splits


def set_seeds(seed: int = config.SEED) -> None:
    """Fija semillas de Python, NumPy y TensorFlow.

    Límite: en GPU o con distinto hardware/versión de librerías los
    resultados pueden variar ligeramente aunque la semilla sea la misma.
    """
    from tensorflow import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)       # random, numpy y tensorflow
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # versiones antiguas no lo soportan
        pass


def train(max_epochs: int = config.MAX_EPOCHS, verbose: int = 2) -> dict:
    """Entrena el modelo y guarda artefactos. Devuelve un resumen."""
    if not isinstance(max_epochs, int) or isinstance(max_epochs, bool) or max_epochs < 1:
        raise ValueError("max_epochs debe ser un entero mayor que cero.")
    config.ensure_dirs()
    set_seeds()

    import tensorflow as tf
    from tensorflow import keras
    from .model import build_mlp

    splits = load_splits()
    model = build_mlp()
    model.summary()

    callbacks = [
        # Detiene el entrenamiento si la pérdida de validación deja de mejorar
        # y restaura los pesos de la mejor época.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        splits.x_train, splits.y_train,
        validation_data=(splits.x_val, splits.y_val),
        epochs=max_epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start

    model.save(config.MODEL_PATH)

    hist = {k: [float(v) for v in values] for k, values in history.history.items()}
    config.HISTORY_PATH.write_text(json.dumps(hist, indent=2), encoding="utf-8")

    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    summary = {
        "epochs_run": len(hist["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": hist["val_loss"][best_epoch - 1],
        "best_val_accuracy": hist["val_accuracy"][best_epoch - 1],
        "train_loss_at_best": hist["loss"][best_epoch - 1],
        "train_accuracy_at_best": hist["accuracy"][best_epoch - 1],
        "training_seconds": round(elapsed, 1),
        "trainable_params": int(model.count_params()),
        "samples": {
            "train": int(len(splits.y_train)),
            "validation": int(len(splits.y_val)),
            "test": int(len(splits.y_test)),
        },
        "class_counts": {
            "train": np.bincount(splits.y_train, minlength=10).tolist(),
            "validation": np.bincount(splits.y_val, minlength=10).tolist(),
            "test": np.bincount(splits.y_test, minlength=10).tolist(),
        },
        "hyperparameters": {
            "hidden_units": list(config.HIDDEN_UNITS),
            "dropout": config.DROPOUT_RATE,
            "optimizer": "Adam",
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "max_epochs": max_epochs,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "seed": config.SEED,
        },
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    config.TRAIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nModelo guardado en: {config.MODEL_PATH}")
    print(f"Mejor época: {best_epoch} | val_loss={summary['best_val_loss']:.4f} "
          f"| val_acc={summary['best_val_accuracy']:.4f} | tiempo={elapsed:.1f}s")
    return summary
