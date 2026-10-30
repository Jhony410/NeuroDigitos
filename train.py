"""Entrena el MLP sobre MNIST y lo guarda en models/mlp_mnist.keras.

Uso:
    python train.py              # entrena (máx. 50 épocas con EarlyStopping)
    python train.py --epochs 5   # prueba rápida
    python train.py --evaluate   # entrena y luego evalúa en prueba
"""
import argparse

from digit_recognizer import config
from digit_recognizer.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento del MLP para dígitos 0-9")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS,
                        help="número máximo de épocas")
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluar en el conjunto de prueba al terminar")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs debe ser mayor que cero")

    train(max_epochs=args.epochs)
    if args.evaluate:
        from digit_recognizer.evaluation import evaluate
        evaluate()


if __name__ == "__main__":
    main()
