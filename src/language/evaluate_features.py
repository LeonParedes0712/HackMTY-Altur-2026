"""Compara tres variantes fijas sobre train/val, sin guardar ni sustituir modelos."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .build_features import FEATURES
except ImportError:
    from build_features import FEATURES

ROOT = Path(__file__).resolve().parents[2]
ACOUSTIC = [f"mfcc_{i:02d}_{s}" for i in range(1, 14) for s in ("mean", "std")]
ACOUSTIC += ["pitch_mean", "pitch_std", "voiced_fraction", "rms_mean", "rms_std"]


def evaluate(root, out):
    manifest = pd.read_csv(root / "manifest.csv")
    language = pd.read_csv(out / "language_features_corrected.csv")
    acoustic = pd.read_csv(root / "outputs/acoustic_features.csv")
    for name, frame in (("lenguaje", language), ("acústica", acoustic)):
        if frame.anon_id.duplicated().any() or set(frame.anon_id) != set(manifest.anon_id):
            raise ValueError(f"Cobertura o IDs inválidos en {name}")
    data = manifest[["anon_id", "label", "split"]].merge(
        language[["anon_id", *FEATURES]], on="anon_id", validate="one_to_one").merge(
        acoustic[["anon_id", *ACOUSTIC]], on="anon_id", validate="one_to_one")
    if not data.label.isin(["human", "synthetic"]).all() or not data.split.isin(["train", "val"]).all():
        raise ValueError("Clases o particiones desconocidas")
    train, val = data[data.split.eq("train")], data[data.split.eq("val")]
    for frame in (train, val):
        if set(frame.label) != {"human", "synthetic"}:
            raise ValueError("Cada partición debe contener ambas clases")
    ytrain, yval = train.label.eq("synthetic").astype(int), val.label.eq("synthetic").astype(int)
    results, predictions = {}, val[["anon_id", "label"]].copy()
    variants = {"acoustic": ACOUSTIC, "language": FEATURES, "combined": ACOUSTIC + FEATURES}
    for name, columns in variants.items():
        if np.isinf(data[columns].to_numpy(dtype=float)).any() or train[columns].isna().all().any():
            raise ValueError(f"Características inválidas: {name}")
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=2000, random_state=42))
        model.fit(train[columns], ytrain)
        probability = model.predict_proba(val[columns])[:, list(model.classes_).index(1)]
        pred = probability >= 0.5
        matrix = confusion_matrix(yval, pred, labels=[0, 1])
        results[name] = {"accuracy": accuracy_score(yval, pred), "f1_synthetic": f1_score(yval, pred),
                         "log_loss": log_loss(yval, probability), "confusion_matrix": matrix.tolist(),
                         "errors": int((pred != yval).sum()), "feature_count": len(columns)}
        predictions[f"{name}_p_synthetic"] = probability
        predictions[f"{name}_correct"] = pred == yval
    report = {"train_calls": len(train), "validation_calls": len(val), "threshold": 0.5,
              "protocol": "Tres variantes fijadas previamente; imputación y escalado ajustados solo con train.",
              "confusion_matrix_order": ["human", "synthetic"], "results": results,
              "limitation": "Comparación exploratoria en val ya observado; no demuestra desempeño en voces ocultas."}
    (out / "model_comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluate(args.root, args.output_dir or args.root / "review/language")


if __name__ == "__main__":
    main()
