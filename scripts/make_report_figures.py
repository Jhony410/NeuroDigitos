"""Genera figuras auxiliares para el informe:
- results/figures/pipeline_preprocesamiento.png  (pasos del preprocesamiento)
- results/figures/montaje_gui_0_9.png            (capturas de los dígitos 0-9)

Uso:  python scripts/make_report_figures.py   (después de capture_screenshots.py)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from capture_screenshots import DIGITS, catmull  # noqa: E402,F401
from digit_recognizer.preprocessing import DIGIT_BOX, INK_THRESHOLD, preprocess_canvas  # noqa: E402

FIG = ROOT / "results" / "figures"
SHOTS = ROOT / "results" / "screenshots"


def pipeline_figure():
    # Dígito 5 dibujado pequeño y desplazado, para que el efecto sea visible.
    img = Image.new("L", (280, 280), 0)
    d = ImageDraw.Draw(img)
    for stroke in DIGITS[5]:
        pts = [(x * 0.6 + 100, y * 0.6 + 95) for x, y in stroke]
        d.line(pts, fill=255, width=20, joint="curve")
    gray = np.asarray(img)
    ys, xs = np.nonzero(gray > INK_THRESHOLD)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = DIGIT_BOX / max(h, w)
    small = np.asarray(Image.fromarray(crop).resize((max(1, round(w * s)), max(1, round(h * s))),
                                                     Image.Resampling.LANCZOS))
    final = preprocess_canvas(img)[0]

    panels = [(gray, "1. Lienzo 280×280\n(trazo claro, fondo negro)"),
              (crop, f"2. Recorte del área\ncon tinta ({w}×{h})"),
              (small, f"3. Redimensionado\n(lado mayor = 20 px)"),
              (final, "4. 28×28 centrado por\nmasa y normalizado [0,1]")]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    for ax, (arr, title) in zip(axes, panels):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "pipeline_preprocesamiento.png", dpi=170)
    plt.close(fig)


def montage():
    ims = [Image.open(SHOTS / f"04_digito_{d}.png").convert("RGB") for d in range(10)]
    w, h = ims[0].size
    sheet = Image.new("RGB", (w * 5, h * 2), "white")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % 5) * w, (k // 5) * h))
    sheet.save(FIG / "montaje_gui_0_9.png")


if __name__ == "__main__":
    pipeline_figure()
    montage()
    print("Figuras generadas en", FIG)
