"""Reconoce un dígito aislado: python predict.py imagen.png."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from digit_recognizer import config
from digit_recognizer.inference import load_predictor, predict_probabilities
from digit_recognizer.preprocessing import preprocess_canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args()
    try:
        with Image.open(args.image) as image:
            batch = preprocess_canvas(ImageOps.exif_transpose(image))
        probs = predict_probabilities(load_predictor(args.model), batch)[0]
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps({"digit": int(probs.argmax()),
                      "probabilities": [float(p) for p in probs],
                      "scope": "Un dígito aislado; softmax no garantiza un acierto."},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
