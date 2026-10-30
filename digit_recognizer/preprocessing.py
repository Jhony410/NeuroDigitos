"""Convierte un dibujo del lienzo en una entrada compatible con MNIST.

MNIST se construyó así: el dígito se recorta, se escala a una caja de
20x20 conservando la proporción y se coloca en una imagen de 28x28
centrado según su centro de masa. Aquí se replica ese proceso para que
el dibujo del usuario se parezca lo más posible a los datos de entrenamiento.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import config

INK_THRESHOLD = 30        # nivel de gris mínimo para considerar "tinta"
MIN_INK_PIXELS = 20       # menos píxeles que esto = lienzo vacío
DIGIT_BOX = 20            # tamaño máximo del dígito dentro de 28x28 (como MNIST)


class EmptyCanvasError(ValueError):
    """Se lanza cuando el lienzo no contiene un trazo suficiente."""


def is_empty(gray: np.ndarray) -> bool:
    """True si el arreglo en escala de grises no contiene trazo útil."""
    return int((gray > INK_THRESHOLD).sum()) < MIN_INK_PIXELS


def _center_by_mass(img28: np.ndarray) -> np.ndarray:
    """Desplaza la imagen para que su centro de masa quede en el centro."""
    total = img28.sum()
    if total == 0:
        return img28
    rows, cols = np.indices(img28.shape)
    cy = (rows * img28).sum() / total
    cx = (cols * img28).sum() / total
    shift_y = int(round(img28.shape[0] / 2 - cy))
    shift_x = int(round(img28.shape[1] / 2 - cx))

    shifted = np.zeros_like(img28)
    h, w = img28.shape
    src_y = slice(max(0, -shift_y), min(h, h - shift_y))
    dst_y = slice(max(0, shift_y), min(h, h + shift_y))
    src_x = slice(max(0, -shift_x), min(w, w - shift_x))
    dst_x = slice(max(0, shift_x), min(w, w + shift_x))
    shifted[dst_y, dst_x] = img28[src_y, src_x]
    return shifted


def preprocess_canvas(image: Image.Image) -> np.ndarray:
    """Imagen PIL del lienzo (trazo claro sobre fondo oscuro) -> arreglo (1, 28, 28).

    Pasos: escala de grises, recorte del área con tinta, redimensionado
    conservando proporciones, márgenes y centrado, y normalización a [0, 1].
    """
    # PNG transparente: componer sobre blanco antes de convertir a gris.
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.size == 0:
        raise EmptyCanvasError("La imagen no contiene píxeles.")

    # Si el fondo es claro (p. ej. una imagen externa), se invierte.
    if np.median(gray) > 127:
        gray = 255 - gray

    if is_empty(gray):
        raise EmptyCanvasError("El lienzo está vacío o el trazo es demasiado pequeño.")

    # 1) Recorte del rectángulo que contiene la tinta.
    ys, xs = np.nonzero(gray > INK_THRESHOLD)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 2) Redimensionar conservando proporciones: el lado mayor mide 20 px.
    h, w = crop.shape
    scale = DIGIT_BOX / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small = Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 3) Pegar en un lienzo negro de 28x28 (márgenes) y centrar por masa.
    canvas28 = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
    top = (config.IMAGE_SIZE - new_h) // 2
    left = (config.IMAGE_SIZE - new_w) // 2
    canvas28[top:top + new_h, left:left + new_w] = np.asarray(small, dtype=np.float32)
    canvas28 = _center_by_mass(canvas28)

    # 4) Normalizar y añadir la dimensión de lote: (1, 28, 28).
    canvas28 = np.clip(canvas28 / 255.0, 0.0, 1.0)
    return canvas28[np.newaxis, ...].astype("float32")
