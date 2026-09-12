from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "outputs" / "acoustic_features.csv"
MODEL_PATH = ROOT / "models" / "acoustic_logistic.pkl"

LABEL_MAP = {"human": 0, "synthetic": 1}

# Lista explícita para no incluir accidentalmente metadatos.
FEATURE_COLUMNS = [
    f"mfcc_{number:02d}_{stat}"
    for number in range(1, 14)
    for stat in ("mean", "std")
] + [
    "pitch_mean",
    "pitch_std",
    "voiced_fraction",
    "rms_mean",
    "rms_std",
]


def main():
    df = pd.read_csv(
        DATA_PATH,
        dtype={
            "anon_id": "string",
            "label": "string",
            "split": "string",
        },
    )

    required = {"anon_id", "label", "split", *FEATURE_COLUMNS}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en el CSV: {sorted(missing)}. "
            "Revisa build_dataset.py y vuelve a generar el CSV."
        )

    for column in ("anon_id", "label", "split"):
        df[column] = df[column].str.strip()

        if df[column].isna().any() or df[column].eq("").any():
            raise ValueError(f"Hay valores vacíos en {column}")

    df["label"] = df["label"].str.lower()
    df["split"] = df["split"].str.lower()

    if df["anon_id"].duplicated().any():
        raise ValueError(
            "Hay anon_id duplicados. Cada audio debe aparecer una sola vez."
        )

    invalid_labels = set(df["label"]) - set(LABEL_MAP)
    if invalid_labels:
        raise ValueError(f"Etiquetas desconocidas: {sorted(invalid_labels)}")

    invalid_splits = set(df["split"]) - {"train", "val"}
    if invalid_splits:
        raise ValueError(f"Splits desconocidos: {sorted(invalid_splits)}")

    # Aceptamos NaN, pero rechazamos texto e infinitos.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="raise",
    )

    if np.isinf(df[FEATURE_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError("Hay valores infinitos en las características")

    if df[FEATURE_COLUMNS].isna().all(axis=1).any():
        raise ValueError(
            "Hay audios sin ninguna característica válida. "
            "Revisa la extracción antes de entrenar."
        )

    train_df = df.loc[df["split"] == "train"].copy()
    val_df = df.loc[df["split"] == "val"].copy()

    for name, subset in (("train", train_df), ("val", val_df)):
        if subset.empty:
            raise ValueError(f"El conjunto {name} está vacío")

        if set(subset["label"]) != set(LABEL_MAP):
            raise ValueError(
                f"El conjunto {name} debe contener human y synthetic"
            )

    X_train = train_df[FEATURE_COLUMNS]
    X_val = val_df[FEATURE_COLUMNS]

    # No se puede calcular una mediana sin observaciones.
    empty_features = X_train.columns[X_train.isna().all()].tolist()
    if empty_features:
        raise ValueError(
            "Estas características están vacías en todo train: "
            f"{empty_features}"
        )

    y_train = train_df["label"].map(LABEL_MAP).astype(int)
    y_val = val_df["label"].map(LABEL_MAP).astype(int)

    print(f"Train: {len(train_df)} | Validation: {len(val_df)}")
    print(f"Características: {len(FEATURE_COLUMNS)}")
    print("\nDistribución de clases:")
    print(pd.crosstab(df["split"], df["label"]))
    print(
        "\nValores faltantes que se imputarán:",
        int(df[FEATURE_COLUMNS].isna().sum().sum()),
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        ),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    # Imputación, escalado y clasificador se ajustan solo con train.
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    synthetic_index = list(model.classes_).index(1)
    y_prob = model.predict_proba(X_val)[:, synthetic_index]

    print("\nReporte de validación:")
    print(classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["human", "synthetic"],
        zero_division=0,
    ))

    print("Matriz de confusión — filas: real; columnas: predicción")
    print("Orden: human, synthetic")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1]))

    print(
        "\nF1 synthetic:",
        round(f1_score(y_val, y_pred, zero_division=0), 4),
    )
    print(
        "Log loss:",
        round(log_loss(y_val, y_prob, labels=[0, 1]), 4),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "label_map": LABEL_MAP,
    }
    joblib.dump(artifact, MODEL_PATH)

    print("\nModelo guardado en:", MODEL_PATH)


if __name__ == "__main__":
    main()