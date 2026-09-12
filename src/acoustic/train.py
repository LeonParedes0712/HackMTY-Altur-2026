"""Train on train only, report validation, export P(synthetic)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .features import feature_names, extract_call


def main():
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, default=Path('outputs/acoustic_features.csv'))
    p.add_argument('--output', type=Path, default=Path('outputs'))
    p.add_argument('--model', choices=['logistic', 'forest'], default='logistic')
    a = p.parse_args()
    with a.features.open() as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['anon_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate rows')
    if any(r['status'] != 'ok' for r in rows):
        raise ValueError('Resolve failed/no-speech rows before training; do not silently drop calls')
    modes = {r['segmentation'] for r in rows}
    if len(modes) != 1:
        raise ValueError('Mixed segmentation modes')
    names = feature_names()
    X = np.array([[float(r[n]) for n in names] for r in rows])
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite features')
    if any(r['label'] not in ('human', 'synthetic') or r['split'] not in ('train', 'val') for r in rows):
        raise ValueError('Invalid labels or splits')
    y = np.array([int(r['label'] == 'synthetic') for r in rows])
    train = np.array([r['split'] == 'train' for r in rows])
    val = ~train
    if len(np.unique(y[train])) != 2 or len(np.unique(y[val])) != 2:
        raise ValueError('Both classes are required in train and val')
    model = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
             if a.model == 'logistic' else RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1))
    model.fit(X[train], y[train])
    prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
    pred = prob[val] >= 0.5
    truth = y[val]
    metrics = {'model': a.model, 'train_calls': int(train.sum()), 'val_calls': int(val.sum()),
               'accuracy': accuracy_score(truth, pred), 'precision': precision_score(truth, pred, zero_division=0),
               'recall': recall_score(truth, pred, zero_division=0), 'f1': f1_score(truth, pred, zero_division=0),
               'roc_auc': roc_auc_score(truth, prob[val]), 'brier': brier_score_loss(truth, prob[val]),
               'confusion_matrix_human_synthetic': confusion_matrix(truth, pred, labels=[0, 1]).tolist(),
               'note': 'Train scores are in-sample. Never use them to train a fusion model; use out-of-fold scores.'}
    a.output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': names, 'segmentation': modes.pop()}, a.output / 'acoustic_model.joblib')
    (a.output / 'acoustic_metrics.json').write_text(json.dumps(metrics, indent=2))
    with (a.output / 'acoustic_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['anon_id', 'acoustic_score'])
        writer.writerows((r['anon_id'], float(score)) for r, score in zip(rows, prob))
    print(json.dumps(metrics, indent=2))


def predict_call(wav_path, model_path, turns_path=None):
    import joblib
    bundle = joblib.load(model_path)  # Load only trusted team-created artifacts.
    values, diagnostics = extract_call(wav_path, turns_path, bundle['segmentation'])
    if diagnostics['status'] != 'ok':
        raise ValueError('Insufficient caller speech; integration must handle this case')
    model = bundle['model']
    X = np.array([[values[n] for n in bundle['features']]])
    return float(model.predict_proba(X)[0, list(model.classes_).index(1)])

if __name__ == '__main__':
    main()
