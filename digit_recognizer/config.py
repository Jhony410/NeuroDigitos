"""Configuración central del proyecto: rutas e hiperparámetros.

Todos los scripts leen sus valores desde aquí para que el experimento
sea fácil de reproducir y de modificar en un solo lugar.
"""
from pathlib import Path

# --- Rutas -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

LOCAL_MNIST_PATH = DATA_DIR / "mnist.npz"          # copia local opcional
MODEL_PATH = MODELS_DIR / "mlp_mnist.keras"        # modelo entrenado
HISTORY_PATH = RESULTS_DIR / "history.json"        # curvas de entrenamiento
TRAIN_SUMMARY_PATH = RESULTS_DIR / "train_summary.json"
METRICS_PATH = RESULTS_DIR / "test_metrics.json"   # métricas sobre prueba
REPORT_PATH = RESULTS_DIR / "classification_report.txt"

# --- Datos -------------------------------------------------------------------
IMAGE_SIZE = 28
NUM_CLASSES = 10
VALIDATION_FRACTION = 0.10   # 10 % del conjunto de entrenamiento (estratificado)

# --- Reproducibilidad -------------------------------------------------------
SEED = 42

# --- Arquitectura (MLP) -----------------------------------------------------
HIDDEN_UNITS = (128, 64)
DROPOUT_RATE = 0.2

# --- Entrenamiento ----------------------------------------------------------
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5


def ensure_dirs() -> None:
    """Crea las carpetas de salida si no existen."""
    for folder in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
