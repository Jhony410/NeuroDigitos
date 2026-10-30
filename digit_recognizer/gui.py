"""Interfaz gráfica (Tkinter) para dibujar un dígito y reconocerlo.

La aplicación NO entrena: carga el modelo guardado en models/mlp_mnist.keras.
Reconoce UN solo dígito (0-9) por dibujo.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import config
from .preprocessing import EmptyCanvasError, preprocess_canvas
from .inference import load_predictor, predict_probabilities

CANVAS_SIZE = 280          # 10 veces 28 px
BRUSH_RADIUS = 10          # grosor del trazo (≈ 2 px tras reducir a 28x28)
PREVIEW_SIZE = 140         # vista ampliada de la entrada de 28x28
BAR_WIDTH = 220

BG = "#f4f5f7"
ACCENT = "#1f6feb"
OK = "#1a7f37"
WARN = "#b35900"
ERR = "#cf222e"


def _pick_family(candidates: tuple[str, ...]) -> str:
    """Devuelve la primera fuente instalada de la lista (o la de Tk por defecto)."""
    installed = {name.lower(): name for name in tkfont.families()}
    for name in candidates:
        if name.lower() in installed:
            return installed[name.lower()]
    return tkfont.nametofont("TkDefaultFont").actual()["family"]


class DigitRecognizerApp:
    def __init__(self, root: tk.Tk, model_path: Path = config.MODEL_PATH,
                 show_dialogs: bool = True, load_async: bool = True) -> None:
        self.root = root
        self.model_path = Path(model_path)
        self.show_dialogs = show_dialogs
        self.model = None
        self.last_input: np.ndarray | None = None
        self._last_xy: tuple[int, int] | None = None

        # Imagen "espejo" en memoria: lo que se dibuja en el lienzo también
        # se dibuja aquí, y es lo que se envía al preprocesamiento.
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        self._build_ui()
        if load_async:
            self.root.after(150, self.load_model)
        else:
            self.load_model()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        # Fuentes disponibles en el sistema (Windows: Segoe UI / Consolas).
        self.sans = _pick_family(("Segoe UI", "DejaVu Sans", "Helvetica", "Arial"))
        self.mono = _pick_family(("Consolas", "DejaVu Sans Mono", "Courier New", "Courier"))
        self.root.title("Reconocedor de dígitos 0–9 · MLP (MNIST)")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        title = tk.Label(self.root, text="Reconocimiento de dígitos manuscritos",
                         font=(self.sans, 15, "bold"), bg=BG)
        title.grid(row=0, column=0, columnspan=2, pady=(12, 2))
        subtitle = tk.Label(self.root, bg=BG, fg="#555",
                            text="Dibuje UN dígito (0–9) con el mouse y presione «Reconocer».",
                            font=(self.sans, 10))
        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 8))

        # --- Panel izquierdo: lienzo y botones ---
        left = tk.Frame(self.root, bg=BG)
        left.grid(row=2, column=0, padx=(16, 8), sticky="n")
        tk.Label(left, text="Lienzo (280 × 280)", bg=BG,
                 font=(self.sans, 10, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black",
                                highlightthickness=2, highlightbackground="#888",
                                cursor="pencil")
        self.canvas.pack(pady=(2, 8))
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        buttons = tk.Frame(left, bg=BG)
        buttons.pack(fill="x")
        self.btn_predict = tk.Button(buttons, text="Reconocer  (Enter)", width=16,
                                     command=self.recognize, bg="#b8c0cc", fg="white",
                                     activebackground="#1158c7", activeforeground="white",
                                     font=(self.sans, 10, "bold"), state="disabled")
        self.btn_predict.pack(side="left", padx=(0, 6))
        self.btn_clear = tk.Button(buttons, text="Limpiar  (Esc)", width=14,
                                   command=self.clear, font=(self.sans, 10))
        self.btn_clear.pack(side="left")
        self.root.bind("<Return>", lambda _e: self.recognize())
        self.root.bind("<Escape>", lambda _e: self.clear())

        # --- Panel derecho: resultado, probabilidades y vista 28x28 ---
        right = tk.Frame(self.root, bg=BG)
        right.grid(row=2, column=1, padx=(8, 16), sticky="n")

        res = tk.Frame(right, bg="white", highlightthickness=1, highlightbackground="#ccc")
        res.pack(fill="x")
        tk.Label(res, text="Dígito predicho", bg="white",
                 font=(self.sans, 10, "bold")).pack(pady=(6, 0))
        self.lbl_digit = tk.Label(res, text="–", bg="white", fg=ACCENT,
                                  font=(self.sans, 44, "bold"))
        self.lbl_digit.pack()
        self.lbl_conf = tk.Label(res, text="Confianza (softmax): –", bg="white",
                                 font=(self.sans, 10))
        self.lbl_conf.pack(pady=(0, 6))

        tk.Label(right, text="Probabilidades por clase", bg=BG,
                 font=(self.sans, 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.bars = tk.Canvas(right, width=BAR_WIDTH + 80, height=10 * 19 + 6,
                              bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.bars.pack()
        self._draw_bars(np.zeros(config.NUM_CLASSES))

        preview_row = tk.Frame(right, bg=BG)
        preview_row.pack(fill="x", pady=(8, 0))
        tk.Label(preview_row, text="Entrada del modelo\n(28 × 28 ampliada)", bg=BG,
                 justify="left", font=(self.sans, 9, "bold")).pack(side="left", anchor="n")
        self.preview = tk.Label(preview_row, bg="black", width=PREVIEW_SIZE // 2,
                                height=PREVIEW_SIZE // 2)
        self.preview.pack(side="right")
        self._set_preview(None)

        # --- Barra de estado ---
        self.status = tk.Label(self.root, text="Iniciando…", anchor="w", bg="#e6e8eb",
                               font=(self.sans, 9), padx=8, pady=4)
        self.status.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))

    # ------------------------------------------------------------- Modelo --
    def load_model(self) -> bool:
        """Carga el modelo .keras. Si no existe, informa y deshabilita «Reconocer»."""
        if not self.model_path.exists():
            self.model = None
            self._set_predict_enabled(False)
            self.notify(
                f"No se encontró el modelo «{self.model_path.name}». "
                "Ejecute primero:  python train.py", "error",
                dialog_title="Modelo no encontrado")
            return False

        self.notify("Cargando modelo…", "info")
        self.root.update_idletasks()
        try:
            self.model = load_predictor(self.model_path)
        except Exception as exc:  # archivo dañado o incompatible
            self.model = None
            self._set_predict_enabled(False)
            self.notify(f"No se pudo cargar el modelo: {exc}", "error",
                        dialog_title="Error al cargar el modelo")
            return False

        self._set_predict_enabled(True)
        self.notify(f"Modelo cargado: {self.model_path.name}. Dibuje un dígito.", "ok")
        return True

    # ------------------------------------------------------------- Dibujo --
    def _paint_dot(self, x: int, y: int) -> None:
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _on_press(self, event: tk.Event) -> None:
        self._last_xy = (event.x, event.y)
        self._paint_dot(event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self._last_xy is None:
            self._last_xy = (event.x, event.y)
        x0, y0 = self._last_xy
        width = 2 * BRUSH_RADIUS
        self.canvas.create_line(x0, y0, event.x, event.y, fill="white", width=width,
                                capstyle="round", smooth=True)
        self.draw.line([x0, y0, event.x, event.y], fill=255, width=width)
        self._paint_dot(event.x, event.y)
        self._last_xy = (event.x, event.y)

    def _on_release(self, _event: tk.Event) -> None:
        self._last_xy = None

    def clear(self) -> None:
        """Borra el lienzo, la imagen espejo y los resultados."""
        self.canvas.delete("all")
        self._last_xy = None
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.last_input = None
        self.lbl_digit.configure(text="–", fg=ACCENT)
        self.lbl_conf.configure(text="Confianza (softmax): –")
        self._draw_bars(np.zeros(config.NUM_CLASSES))
        self._set_preview(None)
        if self.model is not None:
            self.notify("Lienzo limpio. Dibuje un nuevo dígito.", "info")

    # --------------------------------------------------------- Predicción --
    def recognize(self) -> tuple[int, np.ndarray] | None:
        """Preprocesa el dibujo, predice y actualiza la interfaz."""
        if self.model is None:
            self.notify("No hay un modelo cargado. Ejecute primero:  python train.py",
                        "error", dialog_title="Modelo no disponible")
            return None
        try:
            x = preprocess_canvas(self.image)
        except EmptyCanvasError:
            self.notify("El lienzo está vacío. Dibuje un dígito antes de reconocer.",
                        "warning", dialog_title="Lienzo vacío")
            return None

        try:
            probs = predict_probabilities(self.model, x)[0]
        except Exception as exc:
            self.clear()
            self.notify(f"No se pudo reconocer el dibujo: {exc}", "error",
                        dialog_title="Error de predicción")
            return None
        self.last_input = x
        digit = int(np.argmax(probs))

        self.lbl_digit.configure(text=str(digit), fg=OK)
        self.lbl_conf.configure(text=f"Confianza (softmax): {probs[digit] * 100:.1f} %")
        self._draw_bars(probs)
        self._set_preview(x[0])
        self.notify(f"Predicción: {digit}  (P = {probs[digit]:.3f}). "
                    "Una probabilidad alta no garantiza que sea correcta.", "ok")
        return digit, probs

    # ---------------------------------------------------------- Auxiliares --
    def _set_predict_enabled(self, enabled: bool) -> None:
        self.btn_predict.configure(state="normal" if enabled else "disabled",
                                   bg=ACCENT if enabled else "#b8c0cc")

    def _draw_bars(self, probs: np.ndarray) -> None:
        self.bars.delete("all")
        best = int(np.argmax(probs)) if probs.sum() > 0 else -1
        for d, p in enumerate(probs):
            y = 6 + d * 19
            self.bars.create_text(14, y + 8, text=str(d), font=(self.sans, 10, "bold"))
            self.bars.create_rectangle(28, y + 2, 28 + BAR_WIDTH, y + 15,
                                       fill="#eef0f3", outline="")
            if p > 0:
                color = OK if d == best else "#8aa4c8"
                self.bars.create_rectangle(28, y + 2, 28 + max(1, p * BAR_WIDTH), y + 15,
                                           fill=color, outline="")
            self.bars.create_text(28 + BAR_WIDTH + 6, y + 8, anchor="w",
                                  text=f"{p * 100:5.1f}%", font=(self.mono, 9))

    def _set_preview(self, img28: np.ndarray | None) -> None:
        if img28 is None:
            arr = np.zeros((28, 28), dtype=np.uint8)
        else:
            arr = (np.clip(img28, 0, 1) * 255).astype(np.uint8)
        big = Image.fromarray(arr).resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(big)  # referencia para que no se libere
        self.preview.configure(image=self._preview_photo, width=PREVIEW_SIZE,
                               height=PREVIEW_SIZE)

    def notify(self, message: str, level: str = "info", dialog_title: str | None = None) -> None:
        """Muestra el mensaje en la barra de estado y, si es error/aviso, en un diálogo."""
        colors = {"info": "#e6e8eb", "ok": "#dafbe1", "warning": "#fff1c2", "error": "#ffd7d5"}
        fg = {"info": "#24292f", "ok": OK, "warning": WARN, "error": ERR}
        self.status.configure(text=message, bg=colors.get(level, "#e6e8eb"),
                              fg=fg.get(level, "#24292f"))
        if self.show_dialogs and dialog_title:
            if level == "error":
                messagebox.showerror(dialog_title, message, parent=self.root)
            elif level == "warning":
                messagebox.showwarning(dialog_title, message, parent=self.root)


def run() -> None:
    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()
