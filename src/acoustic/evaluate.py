"""Evalúa el modelo guardado con val; no reentrena ni cambia el umbral."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

try:
    from .predict import AcousticPredictor
except ImportError:
    from predict import AcousticPredictor

ROOT = Path(__file__).resolve().parents[2]


def main():
    predictor = AcousticPredictor()
    features = pd.read_csv(ROOT / "outputs/acoustic_features.csv")
    manifest = pd.read_csv(ROOT / "manifest.csv")
    data = manifest.merge(features.drop(columns=["label", "split"]),
                          on="anon_id", how="left", validate="one_to_one", indicator=True)
    if not data["_merge"].eq("both").all():
        raise ValueError("Faltan características de llamadas del manifest")
    val = data.loc[data["split"].eq("val")].copy()
    if set(val["label"]) != {"human", "synthetic"}:
        raise ValueError("Validación debe contener ambas clases")
    y = val["label"].eq("synthetic").astype(int)
    probability = predictor.model.predict_proba(val[predictor.columns])[:, predictor.synthetic_index]
    predicted = probability >= 0.5
    val["p_synthetic"] = probability
    val["prediction"] = ["synthetic" if x else "human" for x in predicted]
    val["correct"] = val["prediction"].eq(val["label"])
    report = {
        "source": "saved_model_and_saved_features", "threshold": 0.5,
        "validation_calls": len(val), "accuracy": accuracy_score(y, predicted),
        "f1_synthetic": f1_score(y, predicted), "log_loss": log_loss(y, probability),
        "confusion_matrix_labels": ["human", "synthetic"],
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "errors": val.loc[~val["correct"], ["anon_id", "label", "prediction", "p_synthetic"]].to_dict("records"),
    }
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "acoustic_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    val[["anon_id", "label", "prediction", "p_synthetic", "correct"]].to_csv(out / "acoustic_validation.csv", index=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
