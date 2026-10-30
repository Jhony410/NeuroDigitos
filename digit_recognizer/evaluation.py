"""Evaluación del modelo sobre el conjunto de PRUEBA y generación de figuras.

Este módulo solo se ejecuta después del entrenamiento; sus resultados no
se usan para ajustar hiperparámetros (eso se hizo con validación).
"""
from __future__ import annotations

import json
import hashlib
import platform
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # genera imágenes sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from . import config
from .data import load_splits

LABELS = list(range(config.NUM_CLASSES))
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def plot_learning_curves(history: dict) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = int(np.argmin(history["val_loss"])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in ((axes[0], "loss", "Pérdida (entropía cruzada)"),
                           (axes[1], "accuracy", "Exactitud")):
        ax.plot(epochs, history[key], marker="o", ms=3, label="Entrenamiento")
        ax.plot(epochs, history[f"val_{key}"], marker="s", ms=3, label="Validación")
        ax.axvline(best, color="gray", ls="--", lw=1, label=f"Mejor época ({best})")
        ax.set_xlabel("Época")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "curvas_aprendizaje.png")
    plt.close(fig)


def plot_dataset_overview(x_train: np.ndarray, y_train: np.ndarray,
                          counts: dict) -> None:
    # Ejemplos por clase
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.4))
    for digit in range(10):
        idx = np.flatnonzero(y_train == digit)[:2]
        for row in range(2):
            ax = axes[row, digit]
            ax.imshow(x_train[idx[row]], cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(str(digit))
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "ejemplos_mnist.png")
    plt.close(fig)

    # Distribución de clases
    fig, ax = plt.subplots(figsize=(8, 3.2))
    width = 0.27
    for i, (name, label) in enumerate((("train", "Entrenamiento"),
                                       ("validation", "Validación"),
                                       ("test", "Prueba"))):
        ax.bar(np.arange(10) + (i - 1) * width, counts[name], width, label=label)
    ax.set_xticks(range(10))
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Cantidad de imágenes")
    ax.set_title("Distribución de clases por partición")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "distribucion_clases.png")
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_xlabel("Dígito predicho")
    ax.set_ylabel("Dígito real")
    ax.set_title("Matriz de confusión – conjunto de prueba")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "matriz_confusion.png")
    plt.close(fig)
    return cm


def plot_examples(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                  probs: np.ndarray, correct: bool, filename: str, n: int = 12) -> None:
    mask = (y_true == y_pred) if correct else (y_true != y_pred)
    idx = np.flatnonzero(mask)
    if not correct:
        # Errores más "seguros" primero: muestran los límites de la confianza softmax.
        idx = idx[np.argsort(-probs[idx].max(axis=1))]
    idx = idx[:n]
    fig, axes = plt.subplots(2, 6, figsize=(10, 4.9))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x[i], cmap="gray")
        color = "green" if correct else "red"
        ax.set_title(f"Real: {y_true[i]} | Pred: {y_pred[i]}\nP={probs[i].max():.2f}",
                     fontsize=8.5, color=color)
        ax.axis("off")
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    title = "Predicciones correctas" if correct else "Predicciones incorrectas (mayor confianza)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename)
    plt.close(fig)


def evaluate() -> dict:
    from tensorflow import keras

    config.ensure_dirs()
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {config.MODEL_PATH}. Ejecute primero: python train.py")

    model = keras.models.load_model(config.MODEL_PATH)
    splits = load_splits()

    test_loss, test_acc = model.evaluate(splits.x_test, splits.y_test, verbose=0)
    probs = model.predict(splits.x_test, batch_size=512, verbose=0)
    y_pred = probs.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        splits.y_test, y_pred, labels=LABELS, zero_division=0)
    report_txt = classification_report(splits.y_test, y_pred, labels=LABELS, digits=4)
    config.REPORT_PATH.write_text(report_txt, encoding="utf-8")

    # El historial es opcional: un modelo exportado se puede evaluar por sí solo.
    if config.HISTORY_PATH.exists():
        history = json.loads(config.HISTORY_PATH.read_text(encoding="utf-8"))
        plot_learning_curves(history)
    counts = {name: np.bincount(y, minlength=10).tolist() for name, y in
              (("train", splits.y_train), ("validation", splits.y_val),
               ("test", splits.y_test))}
    plot_dataset_overview(splits.x_train, splits.y_train, counts)
    cm = plot_confusion(splits.y_test, y_pred)
    plot_examples(splits.x_test, splits.y_test, y_pred, probs, True, "ejemplos_correctos.png")
    plot_examples(splits.x_test, splits.y_test, y_pred, probs, False, "ejemplos_incorrectos.png")

    # Pares de confusión más frecuentes (fuera de la diagonal)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top_pairs = []
    for flat in np.argsort(-off, axis=None)[:6]:
        r, c = np.unravel_index(flat, off.shape)
        top_pairs.append({"real": int(r), "predicho": int(c), "cantidad": int(off[r, c])})

    wrong = y_pred != splits.y_test
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(), "numpy": np.__version__},
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "errors": int(wrong.sum()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            str(d): {"precision": float(precision[d]), "recall": float(recall[d]),
                     "f1": float(f1[d]), "support": int(support[d])}
            for d in LABELS
        },
        "top_confusions": top_pairs,
        "mean_confidence_correct": float(probs[~wrong].max(axis=1).mean()) if (~wrong).any() else None,
        "mean_confidence_wrong": float(probs[wrong].max(axis=1).mean()) if wrong.any() else None,
        "wrong_with_confidence_over_0_9": int((probs[wrong].max(axis=1) > 0.9).sum()),
        "confusion_matrix": cm.tolist(),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(report_txt)
    print(f"Pérdida en prueba: {test_loss:.4f} | Exactitud en prueba: {test_acc:.4f}")
    print(f"Figuras guardadas en: {config.FIGURES_DIR}")
    return metrics
