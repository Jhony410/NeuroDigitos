"""Evalúa el modelo guardado sobre el conjunto de prueba oficial de MNIST.

Genera métricas (results/*.json, *.txt) y figuras (results/figures/*.png).

Uso:
    python evaluate.py
"""
from digit_recognizer.evaluation import evaluate

if __name__ == "__main__":
    evaluate()
