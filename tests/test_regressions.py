"""Pruebas sin dependencias adicionales: python -m unittest discover -s tests -p test_regressions.py -v."""
import tempfile
import unittest
from pathlib import Path
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw

from digit_recognizer import config
from digit_recognizer.inference import load_predictor, predict_probabilities
from digit_recognizer.preprocessing import preprocess_canvas, EmptyCanvasError
from digit_recognizer.training import train


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_predictor()

    def test_saved_model_all_classes(self):
        with np.load(config.LOCAL_MNIST_PATH) as data:
            x, y = data['x_test'], data['y_test']
        # Veinte ejemplos de cada clase; selección fija, sin buscar aciertos.
        indices = np.concatenate([np.flatnonzero(y == d)[:20] for d in range(10)])
        probs = predict_probabilities(self.model, x[indices].astype('float32') / 255)
        pred = probs.argmax(axis=1)
        for d in range(10):
            self.assertGreaterEqual(float((pred[y[indices] == d] == d).mean()), 0.8)

    def test_empty_black_white_and_transparent(self):
        for image in (Image.new('L', (280, 280), 0), Image.new('L', (280, 280), 255),
                      Image.new('RGBA', (280, 280), (0, 0, 0, 0))):
            with self.assertRaises(EmptyCanvasError):
                preprocess_canvas(image)

    def test_transparent_black_stroke_matches_white_background(self):
        image = Image.new('RGBA', (280, 280), (0, 0, 0, 0))
        ImageDraw.Draw(image).line([(140, 30), (140, 240)], fill=(0, 0, 0, 255), width=20)
        opaque = Image.new('RGBA', image.size, 'white')
        opaque = Image.alpha_composite(opaque, image)
        np.testing.assert_array_equal(preprocess_canvas(image), preprocess_canvas(opaque))

    def test_invalid_input_and_output(self):
        for x in (np.zeros((28, 28)), np.full((1, 28, 28), np.nan),
                  np.full((1, 28, 28), 255)):
            with self.assertRaises(ValueError):
                predict_probabilities(self.model, x)
        with self.assertRaises(ValueError):
            predict_probabilities(lambda x, training: np.ones((len(x), 10)),
                                  np.zeros((1, 28, 28)))

    def test_invalid_epochs_do_not_train(self):
        for epochs in (0, -1, True):
            with self.assertRaises(ValueError):
                train(epochs)

    def test_incompatible_model_is_rejected(self):
        from tensorflow import keras
        with tempfile.TemporaryDirectory(dir=config.RESULTS_DIR) as folder:
            path = Path(folder) / 'wrong.keras'
            keras.Sequential([keras.Input((4,)), keras.layers.Dense(2)]).save(path)
            with self.assertRaises(ValueError):
                load_predictor(path)

    def test_gui_recognition_clear_and_model_error(self):
        from digit_recognizer.gui import DigitRecognizerApp
        root = tk.Tk()
        root.withdraw()
        try:
            app = DigitRecognizerApp(root, show_dialogs=False, load_async=False)
            self.assertIsNone(app.recognize())
            app.draw.line([(140, 40), (140, 240)], fill=255, width=20)
            self.assertIsNotNone(app.recognize())
            app.clear()
            self.assertIsNone(app.last_input)
            self.assertEqual(np.asarray(app.image).max(), 0)
            app.draw.line([(140, 40), (140, 240)], fill=255, width=20)
            app.model = lambda x, training: np.ones((1, 10))
            self.assertIsNone(app.recognize())
            self.assertIn('No se pudo reconocer', app.status['text'])
            app.model_path = config.MODELS_DIR / 'missing-test.keras'
            self.assertFalse(app.load_model())
            self.assertEqual(str(app.btn_predict['state']), 'disabled')
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main()
