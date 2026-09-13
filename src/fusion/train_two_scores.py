from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "manifest.csv"

ACOUSTIC_FEATURES_PATH = ROOT / "outputs" / "acoustic_features.csv"
BEHAVIOR_FEATURES_PATH = ROOT / "outputs" / "behavior_features.csv"
LANGUAGE_FEATURES_PATH = ROOT / "outputs" / "language_features.csv"

ACOUSTIC_MODEL_PATH = ROOT / "models" / "acoustic_logistic.pkl"
BEHAVIOR_MODEL_PATH = ROOT / "outputs" / "behavior_model.joblib"
LANGUAGE_MODEL_PATH = ROOT / "models" / "language_model.joblib"

FUSION_MODEL_PATH = ROOT / "models" / "fusion_acoustic_behavior.joblib"
PREDICTIONS_PATH = ROOT / "outputs" / "fusion_acoustic_behavior_val_predictions.csv"


def normalize_ids(df):
    df = df.copy()
    df["anon_id"] = (
        df["anon_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.wav$", "", regex=True)
    )
    return df


def align_features(df, ids, columns, name):
    df = normalize_ids(df)

    if df["anon_id"].duplicated().any():
        raise ValueError(f"Hay anon_id duplicados en {name}")

    indexed = df.set_index("anon_id")

    missing = set(ids) - set(indexed.index)
    if missing:
        raise ValueError(
            f"Faltan {len(missing)} IDs en las características {name}"
        )

    return indexed.loc[ids, columns].reset_index(drop=True)


def synthetic_probability(model, X):
    synthetic_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, synthetic_index]


def build_meta_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = normalize_ids(manifest)

    manifest["label"] = (
        manifest["label"].astype(str).str.strip().str.lower()
    )
    manifest["split"] = (
        manifest["split"].astype(str).str.strip().str.lower()
    )

    train_manifest = manifest.loc[
        manifest["split"] == "train"
    ].reset_index(drop=True)

    val_manifest = manifest.loc[
        manifest["split"] == "val"
    ].reset_index(drop=True)

    train_ids = train_manifest["anon_id"].tolist()
    val_ids = val_manifest["anon_id"].tolist()

    y_train = (
        train_manifest["label"] == "synthetic"
    ).astype(int).to_numpy()

    y_val = (
        val_manifest["label"] == "synthetic"
    ).astype(int).to_numpy()

    artifacts = {
        "acoustic": joblib.load(ACOUSTIC_MODEL_PATH),
        "behavior": joblib.load(BEHAVIOR_MODEL_PATH),
    }


    feature_frames = {
        "acoustic": pd.read_csv(ACOUSTIC_FEATURES_PATH),
        "behavior": pd.read_csv(BEHAVIOR_FEATURES_PATH),
    }

    train_features = {}
    val_features = {}

    for name in artifacts:
        columns = artifacts[name]["feature_columns"]

        train_features[name] = align_features(
            feature_frames[name],
            train_ids,
            columns,
            name,
        )

        val_features[name] = align_features(
            feature_frames[name],
            val_ids,
            columns,
            name,
        )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    oof_probabilities = {}
    val_probabilities = {}
    fitted_base_models = {}

    for name, artifact in artifacts.items():
        base_model = artifact["model"]
        X_train = train_features[name]
        X_val = val_features[name]

        oof = np.zeros(len(train_manifest), dtype=float)

        for fit_indices, holdout_indices in cv.split(X_train, y_train):
            fold_model = clone(base_model)

            fold_model.fit(
                X_train.iloc[fit_indices],
                y_train[fit_indices],
            )

            oof[holdout_indices] = synthetic_probability(
                fold_model,
                X_train.iloc[holdout_indices],
            )

        full_model = clone(base_model)
        full_model.fit(X_train, y_train)

        oof_probabilities[name] = oof
        val_probabilities[name] = synthetic_probability(
            full_model,
            X_val,
        )
        fitted_base_models[name] = full_model

        print(
            f"ROC-AUC OOF {name}: "
            f"{roc_auc_score(y_train, oof):.4f}"
        )

    meta_train = pd.DataFrame(oof_probabilities)
    meta_val = pd.DataFrame(val_probabilities)

    candidates = {
        "acoustic_behavior": ["acoustic", "behavior"],
    }

    candidate_scores = {}

    print("\nSelección de modalidades usando solamente train:")

    for candidate_name, columns in candidates.items():
        scores = cross_val_score(
            build_meta_model(),
            meta_train[columns],
            y_train,
            cv=cv,
            scoring="roc_auc",
        )

        candidate_scores[candidate_name] = scores.mean()

        print(
            f"{candidate_name}: "
            f"{scores.mean():.4f} ± {scores.std():.4f}"
        )

    selected_name = max(candidate_scores, key=candidate_scores.get)
    selected_modalities = candidates[selected_name]

    print(f"\nFusión seleccionada: {selected_name}")

    meta_model = build_meta_model()

    meta_oof_probability = cross_val_predict(
        clone(meta_model),
        meta_train[selected_modalities],
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_train,
        meta_oof_probability,
    )

    finite = np.isfinite(thresholds)
    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]

    threshold = float(thresholds[np.argmax(tpr - fpr)])

    meta_model.fit(
        meta_train[selected_modalities],
        y_train,
    )

    fusion_probability = synthetic_probability(
        meta_model,
        meta_val[selected_modalities],
    )

    fusion_prediction = (
        fusion_probability >= threshold
    ).astype(int)

    print(
        "Umbral de fusión obtenido de train:",
        round(threshold, 6),
    )

    print("\nResultados sobre el val oficial:")
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_val, fusion_probability), 4),
    )
    print(
        "Accuracy:",
        round(accuracy_score(y_val, fusion_prediction), 4),
    )
    print(
        "F1 synthetic:",
        round(f1_score(y_val, fusion_prediction), 4),
    )

    print("\nMatriz de confusión:")
    print(confusion_matrix(
        y_val,
        fusion_prediction,
        labels=[0, 1],
    ))

    print("\nReporte:")
    print(classification_report(
        y_val,
        fusion_prediction,
        target_names=["human", "synthetic"],
        digits=4,
        zero_division=0,
    ))

    artifact = {
        "base_models": fitted_base_models,
        "feature_columns": {
            name: artifacts[name]["feature_columns"]
            for name in artifacts
        },
        "meta_model": meta_model,
        "selected_modalities": selected_modalities,
        "threshold": threshold,
        "positive_class": "synthetic",
    }

    FUSION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, FUSION_MODEL_PATH)

    output = val_manifest[
        ["anon_id", "label", "split"]
    ].copy()

    for name, probability in val_probabilities.items():
        output[f"{name}_synthetic_probability"] = probability

    output["fusion_synthetic_probability"] = fusion_probability
    output["fusion_prediction"] = np.where(
        fusion_prediction == 1,
        "synthetic",
        "human",
    )

    output.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nModelo guardado en: {FUSION_MODEL_PATH}")
    print(f"Predicciones guardadas en: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()