"""Lanzador usado SOLO para pruebas automatizadas de la interfaz (capturas).

Abre la misma aplicación que app.py, pero además:
- escribe en un JSON la posición en pantalla del lienzo y de los botones,
- registra cada predicción en un archivo .jsonl.
Así un script externo puede mover el mouse real (xdotool) sobre el lienzo.

Uso:  python scripts/demo_launcher.py <geom.json> <preds.jsonl> [ruta_modelo]
"""
import json
import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digit_recognizer import config  # noqa: E402
from digit_recognizer.gui import DigitRecognizerApp  # noqa: E402

geom_path, preds_path = Path(sys.argv[1]), Path(sys.argv[2])
model_path = Path(sys.argv[3]) if len(sys.argv) > 3 else config.MODEL_PATH

root = tk.Tk()
root.geometry("+40+40")
app = DigitRecognizerApp(root, model_path=model_path)

original = app.recognize


def logged_recognize():
    result = original()
    entry = {"status": app.status["text"]}
    if result is not None:
        digit, probs = result
        entry.update(pred=digit, probs=[round(float(p), 6) for p in probs])
    with preds_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return result


app.recognize = logged_recognize
app.btn_predict.configure(command=logged_recognize)
root.bind("<Return>", lambda _e: logged_recognize())


def dump_geometry():
    root.update_idletasks()

    def box(w):
        return {"x": w.winfo_rootx(), "y": w.winfo_rooty(),
                "w": w.winfo_width(), "h": w.winfo_height()}

    geom = {"window": {"x": root.winfo_rootx(), "y": root.winfo_rooty(),
                       "w": root.winfo_width(), "h": root.winfo_height()},
            "canvas": box(app.canvas), "predict": box(app.btn_predict),
            "clear": box(app.btn_clear),
            "model_loaded": app.model is not None}
    tmp = geom_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(geom), encoding="utf-8")
    tmp.replace(geom_path)   # escritura atómica
    root.after(500, dump_geometry)


CLOSE_FLAG = geom_path.with_suffix(".close_dialog")


def poll_close_dialog():
    """Cierra el cuadro de mensaje abierto cuando el script externo lo pide.

    Los callbacks de after() siguen ejecutándose mientras el diálogo modal
    está abierto, así que se puede pulsar su botón «OK» desde aquí.
    """
    if CLOSE_FLAG.exists():
        CLOSE_FLAG.unlink()
        try:
            root.tk.call(".__tk__messagebox.ok", "invoke")
        except tk.TclError:
            pass
    root.after(200, poll_close_dialog)


root.after(300, dump_geometry)
root.after(300, poll_close_dialog)
root.mainloop()
