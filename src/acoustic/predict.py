import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .features import extract_features
except ImportError:
    from features import extract_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "acoustic_logistic.pkl"


class AcousticPredictor:
    def __init__(self, model_path=MODEL_PATH):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.columns = artifact["feature_columns"]

        # El entrenamiento usa human=0 y synthetic=1.
        classes = list(self.model.classes_)
        if set(classes) != {0, 1}:
            raise ValueError("El modelo debe usar human=0 y synthetic=1")

        self.synthetic_index = classes.index(1)

    def predict(self, audio_path, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("El umbral debe estar entre 0 y 1")

        anon_id = None
        if isinstance(audio_path, (str, Path)):
            audio_path = Path(audio_path)
            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path
            anon_id = audio_path.stem

        values = extract_features(audio_path)

        missing = set(self.columns) - set(values)
        if missing:
            raise ValueError(f"Faltan características: {sorted(missing)}")

        # Conserva exactamente el orden usado al entrenar.
        X = pd.DataFrame([values], columns=self.columns)
        X = X.apply(pd.to_numeric, errors="raise")

        if np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError("Hay características con valores infinitos")

        if X.isna().all(axis=1).any():
            raise ValueError("El audio no tiene características válidas")

        # El Pipeline imputa los NaN y escala automáticamente.
        p_synthetic = float(
            self.model.predict_proba(X)[0, self.synthetic_index]
        )

        if not np.isfinite(p_synthetic) or not 0 <= p_synthetic <= 1:
            raise ValueError("El modelo devolvió una probabilidad inválida")

        return {
            "anon_id": anon_id,
            "p_synthetic": p_synthetic,
            "p_human": 1.0 - p_synthetic,
            "is_synthetic": bool(p_synthetic >= threshold),
            "threshold": threshold,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Estima la probabilidad acústica de voz sintética."
    )
    parser.add_argument("audio", help="Ruta al archivo de audio")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predictor = AcousticPredictor()
    result = predictor.predict(args.audio, threshold=args.threshold)

    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()