from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("outputs/behavior_features.csv")
MODEL_PATH = Path("outputs/behavior_model.joblib")
PREDICTIONS_PATH = Path("outputs/behavior_val_predictions.csv")

FEATURE_COLUMNS = [
    "caller_turn_count",
    "caller_turn_duration_mean",
    "caller_turn_duration_std",
    "response_time_mean",
    "response_time_std",
    "caller_speaking_ratio",
]


def build_models():
    logistic_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )

    forest_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", logistic_preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", forest_preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = (train_df["label"] == "synthetic").astype(int)

    X_val = val_df[FEATURE_COLUMNS]
    y_val = (val_df["label"] == "synthetic").astype(int)

    print(f"Train: {len(train_df)} llamadas")
    print(f"Val:   {len(val_df)} llamadas")

    models = build_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_results = {}

    print("\nROC-AUC mediante validación cruzada dentro de train:")

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        cv_results[name] = scores.mean()

        print(
            f"{name}: "
            f"{scores.mean():.4f} ± {scores.std():.4f}"
        )

    best_name = max(cv_results, key=cv_results.get)
    best_model = models[best_name]

    print(f"\nModelo seleccionado: {best_name}")

    best_model.fit(X_train, y_train)

    val_probability = best_model.predict_proba(X_val)[:, 1]
    val_prediction = (val_probability >= 0.5).astype(int)

    print("\nResultados sobre el split oficial de validación:")
    print(f"ROC-AUC:   {roc_auc_score(y_val, val_probability):.4f}")
    print(f"Accuracy:  {accuracy_score(y_val, val_prediction):.4f}")
    print(f"Precision: {precision_score(y_val, val_prediction):.4f}")
    print(f"Recall:    {recall_score(y_val, val_prediction):.4f}")
    print(f"F1-score:  {f1_score(y_val, val_prediction):.4f}")

    print("\nMatriz de confusión:")
    print("[[human correcto, human clasificado synthetic],")
    print(" [synthetic clasificado human, synthetic correcto]]")
    print(confusion_matrix(y_val, val_prediction, labels=[0, 1]))

    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_val,
            val_prediction,
            target_names=["human", "synthetic"],
            digits=4,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "positive_class": "synthetic",
        "threshold": 0.5,
    }

    joblib.dump(model_bundle, MODEL_PATH)

    predictions = val_df[["anon_id", "label", "split"]].copy()
    predictions["behavior_synthetic_probability"] = val_probability
    predictions["behavior_prediction"] = np.where(
        val_prediction == 1,
        "synthetic",
        "human",
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Predicciones guardadas en: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()