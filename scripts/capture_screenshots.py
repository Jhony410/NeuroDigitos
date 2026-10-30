"""Prueba de extremo a extremo de la interfaz con mouse real (Linux + Xvfb).

Mueve el puntero con xdotool para DIBUJAR los dígitos 0-9 sobre el lienzo,
pulsa los botones y guarda capturas de pantalla con ImageMagick (import).
Los trazos son trayectorias predefinidas que imitan escritura a mano; no
reemplazan las pruebas con dibujos hechos por una persona.

Requisitos (Linux): Xvfb, openbox, xdotool, imagemagick.
Uso:
    Xvfb :99 -screen 0 1280x900x24 &  DISPLAY=:99 openbox &
    DISPLAY=:99 python scripts/capture_screenshots.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
GEOM = Path("/tmp/demo_geom.json")
PREDS = ROOT / "results" / "gui_predictions.jsonl"
PY = sys.executable


def sh(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def ellipse(cx, cy, rx, ry, start=-90, end=270, n=40):
    pts = []
    for i in range(n + 1):
        a = math.radians(start + (end - start) * i / n)
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return pts


def catmull(points, steps=8):
    """Suaviza una polilínea (spline de Catmull-Rom) para imitar un trazo a mano."""
    if len(points) < 3:
        return points
    p = [points[0]] + points + [points[-1]]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for s in range(steps):
            t = s / steps
            t2, t3 = t * t, t * t * t
            out.append(tuple(
                0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t +
                       (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
                       (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3) for k in range(2)))
    out.append(points[-1])
    return out


# Trazos de cada dígito en coordenadas del lienzo (280x280). Lista de trazos.
DIGITS = {
    0: [ellipse(140, 140, 58, 92)],
    1: [catmull([(118, 72), (148, 45), (146, 140), (142, 235)])],
    2: [catmull([(88, 92), (115, 55), (160, 48), (192, 82), (180, 130),
                 (135, 180), (88, 232), (150, 228), (205, 230)])],
    3: [catmull([(85, 70), (135, 45), (188, 72), (178, 118), (135, 135),
                 (190, 165), (185, 215), (135, 238), (82, 215)])],
    4: [catmull([(160, 45), (120, 110), (82, 170), (150, 168), (212, 166)]),
        [(172, 95), (170, 165), (168, 238)]],
    5: [catmull([(198, 52), (150, 50), (102, 52), (96, 128), (150, 112),
                 (195, 148), (188, 205), (140, 236), (86, 214)])],
    6: [catmull([(185, 48), (130, 85), (98, 150), (100, 210), (140, 236),
                 (185, 210), (180, 165), (140, 145), (102, 170)])],
    7: [catmull([(78, 58), (140, 54), (204, 56), (165, 140), (128, 238)])],
    8: [catmull([(180, 70), (140, 45), (98, 72), (110, 115), (175, 160),
                 (190, 210), (140, 240), (90, 210), (105, 160), (170, 115),
                 (182, 72)])],
    9: [ellipse(140, 98, 52, 50, start=0, end=360), catmull([(192, 98), (186, 170), (175, 238)])],
}


# Casos adicionales para analizar límites del sistema.
def _scaled(strokes, scale, dx, dy):
    return [[(x * scale + dx, y * scale + dy) for x, y in st] for st in strokes]


EXTRA_CASES = {
    # Dígito pequeño dibujado en una esquina: prueba recorte y centrado.
    "07_digito_pequeno_esquina": _scaled(DIGITS[3], 0.38, 8, 6),
    # Fuera de alcance: dos dígitos ("12") y una letra ("X").
    "06_fuera_de_alcance_12": (_scaled(DIGITS[1], 0.55, -20, 60)
                               + _scaled(DIGITS[2], 0.55, 115, 60)),
    "06_fuera_de_alcance_X": [[(70, 60), (140, 140), (210, 225)],
                              [(210, 60), (140, 140), (70, 225)]],
}


class GuiSession:
    def __init__(self, model_path: str | None = None):
        GEOM.unlink(missing_ok=True)
        args = [PY, str(ROOT / "scripts" / "demo_launcher.py"), str(GEOM), str(PREDS)]
        if model_path:
            args.append(model_path)
        env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
        self.proc = subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        for _ in range(200):
            if GEOM.exists():
                break
            time.sleep(0.1)
        time.sleep(1.0)

    def geom(self) -> dict:
        for _ in range(20):          # el archivo se reescribe cada 0,5 s
            try:
                return json.loads(GEOM.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                time.sleep(0.05)
        raise RuntimeError("no se pudo leer la geometría de la ventana")

    def wait_model(self, timeout=60):
        for _ in range(timeout * 4):
            if self.geom()["model_loaded"]:
                return
            time.sleep(0.25)
        raise RuntimeError("el modelo no cargó")

    def click(self, name: str):
        b = self.geom()[name]
        sh("xdotool", "mousemove", str(b["x"] + b["w"] // 2), str(b["y"] + b["h"] // 2),
           "click", "1")
        time.sleep(0.8)

    def draw_digit(self, strokes):
        c = self.geom()["canvas"]
        for stroke in strokes:
            x0, y0 = stroke[0]
            sh("xdotool", "mousemove", str(int(c["x"] + x0)), str(int(c["y"] + y0)),
               "mousedown", "1")
            cmd = ["xdotool"]
            for x, y in stroke[1:]:
                cmd += ["mousemove", str(int(c["x"] + x)), str(int(c["y"] + y)), "sleep", "0.01"]
            cmd += ["mouseup", "1"]
            sh(*cmd)
        time.sleep(0.3)

    def shot(self, name: str):
        w = self.geom()["window"]
        pad = 6
        # El diálogo emergente aparece centrado sobre la ventana, por lo que
        # el mismo recorte sirve para ambos casos.
        crop = f"{w['w'] + 2 * pad}x{w['h'] + 30 + pad}+{w['x'] - pad}+{w['y'] - 30}"
        sh("import", "-window", "root", "-crop", crop, "+repage", str(OUT / name))
        print("captura:", name)

    def close_dialog(self):
        """Pide al lanzador que pulse el botón «OK» del diálogo abierto."""
        GEOM.with_suffix(".close_dialog").touch()
        time.sleep(1.0)

    def close(self):
        self.proc.kill()
        self.proc.wait(timeout=10)


def main():
    PREDS.unlink(missing_ok=True)
    GEOM.with_suffix(".close_dialog").unlink(missing_ok=True)

    # 1) Modelo ausente
    s = GuiSession(model_path=str(ROOT / "models" / "no_existe.keras"))
    time.sleep(1.5)
    s.shot("01_modelo_ausente.png")
    s.close_dialog()
    s.shot("01b_modelo_ausente_estado.png")
    s.close()

    # 2) Inicio con modelo cargado
    s = GuiSession()
    s.wait_model()
    time.sleep(0.5)
    s.shot("02_inicio.png")

    # 3) Lienzo vacío
    s.click("predict")
    time.sleep(0.8)
    s.shot("03_lienzo_vacio.png")
    s.close_dialog()

    # 4) Dibujar y reconocer los dígitos 0-9
    for digit, strokes in DIGITS.items():
        s.click("clear")
        s.draw_digit(strokes)
        s.click("predict")
        time.sleep(0.4)
        s.shot(f"04_digito_{digit}.png")

    # 4b) Casos de límite / fuera de alcance
    for name, strokes in EXTRA_CASES.items():
        s.click("clear")
        s.draw_digit(strokes)
        s.click("predict")
        time.sleep(0.4)
        s.shot(f"{name}.png")

    # 5) Botón limpiar
    s.click("clear")
    time.sleep(0.4)
    s.shot("05_despues_de_limpiar.png")
    s.close()


if __name__ == "__main__":
    main()
