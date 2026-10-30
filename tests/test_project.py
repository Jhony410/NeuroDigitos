"""Pruebas automáticas del proyecto.

Ejecutar con:  python -m pytest -v
Las pruebas de la interfaz requieren un entorno gráfico (en Windows funciona
directamente; en Linux sin pantalla use xvfb-run).
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from digit_recognizer import config
from digit_recognizer.data import load_splits
from digit_recognizer.preprocessing import EmptyCanvasError, preprocess_canvas

MODEL_EXISTS = config.MODEL_PATH.exists()


# ------------------------------------------------------------------ Datos --
def test_splits_are_stratified_and_normalized():
    s = load_splits()
    assert s.x_train.shape == (54000, 28, 28)
    assert s.x_val.shape == (6000, 28, 28)
    assert s.x_test.shape == (10000, 28, 28)        # prueba oficial intacta
    assert 0.0 <= s.x_train.min() and s.x_train.max() <= 1.0
    # Estratificación: la proporción de cada clase difiere < 0.5 puntos.
    p_train = np.bincount(s.y_train, minlength=10) / len(s.y_train)
    p_val = np.bincount(s.y_val, minlength=10) / len(s.y_val)
    assert np.abs(p_train - p_val).max() < 0.005


# -------------------------------------------------------- Preprocesamiento --
def _drawn_image(points, width=20):
    img = Image.new("L", (280, 280), 0)
    ImageDraw.Draw(img).line(points, fill=255, width=width)
    return img


def test_empty_canvas_is_detected():
    with pytest.raises(EmptyCanvasError):
        preprocess_canvas(Image.new("L", (280, 280), 0))


def test_preprocessing_shape_range_and_centering():
    # Trazo vertical pegado a la esquina superior izquierda.
    x = preprocess_canvas(_drawn_image([(30, 10), (30, 120)]))
    assert x.shape == (1, 28, 28)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    rows, cols = np.indices((28, 28))
    cy = (rows * x[0]).sum() / x[0].sum()
    cx = (cols * x[0]).sum() / x[0].sum()
    assert abs(cy - 14) < 1.5 and abs(cx - 14) < 1.5   # quedó centrado
    ys, xs = np.nonzero(x[0] > 0.1)
    assert ys.max() - ys.min() <= 21                    # cabe en la caja de 20 px


def test_light_background_is_inverted():
    img = Image.new("L", (280, 280), 255)
    ImageDraw.Draw(img).line([(140, 40), (140, 240)], fill=0, width=20)
    x = preprocess_canvas(img)
    assert x[0].mean() < 0.5      # fondo oscuro, trazo claro (como MNIST)


# ---------------------------------------------------------------- Modelo --
def test_model_architecture():
    from digit_recognizer.model import build_mlp
    model = build_mlp()
    names = [layer.name for layer in model.layers]
    assert names == ["aplanado_784", "oculta_1", "dropout", "oculta_2", "salida_softmax"]
    assert model.output_shape == (None, 10)
    # 784*128+128 + 128*64+64 + 64*10+10 = 109 386 parámetros
    assert model.count_params() == 109_386


@pytest.mark.skipif(not MODEL_EXISTS, reason="ejecute python train.py primero")
def test_saved_model_loads_and_predicts_probabilities():
    from tensorflow import keras
    model = keras.models.load_model(config.MODEL_PATH)
    s = load_splits()
    probs = model.predict(s.x_test[:200], verbose=0)
    assert probs.shape == (200, 10)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert (probs.argmax(axis=1) == s.y_test[:200]).mean() > 0.9


# ------------------------------------------------------------- Interfaz --
@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no hay entorno gráfico disponible")
    root.withdraw()
    yield root
    root.destroy()


def _stroke(app, points):
    c = app.canvas
    c.event_generate("<ButtonPress-1>", x=points[0][0], y=points[0][1])
    for x, y in points[1:]:
        c.event_generate("<B1-Motion>", x=x, y=y)
    c.event_generate("<ButtonRelease-1>", x=points[-1][0], y=points[-1][1])


def test_gui_handles_missing_model(tk_root, tmp_path: Path):
    from digit_recognizer.gui import DigitRecognizerApp
    app = DigitRecognizerApp(tk_root, model_path=tmp_path / "no_existe.keras",
                             show_dialogs=False, load_async=False)
    assert app.model is None
    assert str(app.btn_predict["state"]) == "disabled"
    assert "No se encontró el modelo" in app.status["text"]
    assert app.recognize() is None


@pytest.mark.skipif(not MODEL_EXISTS, reason="ejecute python train.py primero")
def test_gui_empty_canvas_draw_recognize_and_clear(tk_root):
    from digit_recognizer.gui import DigitRecognizerApp
    app = DigitRecognizerApp(tk_root, show_dialogs=False, load_async=False)
    assert app.model is not None

    # Lienzo vacío -> mensaje claro, sin predicción.
    assert app.recognize() is None
    assert "vacío" in app.status["text"]

    # Dibujo de un "1" con eventos de mouse reales.
    tk_root.deiconify()
    tk_root.update()
    _stroke(app, [(140, 40 + 10 * i) for i in range(21)])
    tk_root.update()
    result = app.recognize()
    assert result is not None
    digit, probs = result
    assert probs.shape == (10,) and abs(probs.sum() - 1) < 1e-4
    assert app.lbl_digit["text"] == str(digit)
    assert app.last_input.shape == (1, 28, 28)

    # Limpiar -> imagen espejo en negro y resultados reiniciados.
    app.clear()
    assert np.asarray(app.image).max() == 0
    assert app.lbl_digit["text"] == "–"
    assert app.last_input is None
    assert app.recognize() is None
