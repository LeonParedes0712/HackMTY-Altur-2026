import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from .features import (
    load_turns,
    get_caller_turns,
    get_turn_durations,
    get_response_times,
    get_caller_speaking_ratio,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "outputs" / "behavior_model.joblib"


class BehaviorPredictor:
    def __init__(self, model_path=MODEL_PATH):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.columns = artifact["feature_columns"]
        self.threshold = float(artifact.get("threshold", 0.5))

        classes = list(self.model.classes_)
        if set(classes) != {0, 1}:
            raise ValueError("Se requieren las clases human=0 y synthetic=1")

        self.synthetic_index = classes.index(1)

    def predict(self, audio_path, turns_path):
        info = sf.info(str(audio_path))
        duration = info.frames / info.samplerate

        if duration <= 0:
            raise ValueError("El audio está vacío")

        turns = load_turns(turns_path)
        if not isinstance(turns, list) or not turns:
            raise ValueError("El JSON no contiene turnos")

        previous_start = -1.0
        for turn in turns:
            start = float(turn["start"])
            end = float(turn["end"])

            if turn["channel"] not in (0, 1):
                raise ValueError("Los canales deben ser 0 o 1")

            if not np.isfinite([start, end]).all():
                raise ValueError("Hay tiempos inválidos")

            if not 0 <= start < end <= duration + 0.1:
                raise ValueError("Hay turnos fuera de la duración del audio")

            if start < previous_start:
                raise ValueError("Los turnos deben estar ordenados por inicio")

            turn["start"] = start
            turn["end"] = end
            previous_start = start

        durations = get_turn_durations(get_caller_turns(turns))
        responses = get_response_times(turns)

        if not durations:
            raise ValueError("No hay turnos del caller en el canal 0")

        values = {
            "caller_turn_count": len(durations),
            "caller_turn_duration_mean": float(np.mean(durations)),
            "caller_turn_duration_std": float(np.std(durations)),
            "response_time_mean": (
                float(np.mean(responses)) if responses else np.nan
            ),
            "response_time_std": (
                float(np.std(responses)) if responses else np.nan
            ),
            "caller_speaking_ratio": get_caller_speaking_ratio(
                durations, duration
            ),
        }

        X = pd.DataFrame([values], columns=self.columns)
        probability = float(
            self.model.predict_proba(X)[0, self.synthetic_index]
        )

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("El modelo devolvió una probabilidad inválida")

        return {
            "p_synthetic": probability,
            "p_human": 1.0 - probability,
            "is_synthetic": probability >= self.threshold,
            "threshold": self.threshold,
            "model": "behavior",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("turns")
    args = parser.parse_args()

    result = BehaviorPredictor().predict(args.audio, args.turns)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()