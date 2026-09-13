"""Construye características de las transcripciones corregidas sin alterar originales."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .language_features import extract_language_features
except ImportError:
    from language_features import extract_language_features

ROOT = Path(__file__).resolve().parents[2]
FEATURES = list(extract_language_features("", 1))


def build_features(transcripts, manifest):
    required = {"file_name", "caller_transcript", "status", "duration_sec"}
    if not required.issubset(transcripts.columns):
        raise ValueError("Se requiere el CSV corregido con status y duration_sec")
    transcripts = transcripts.copy()
    transcripts["anon_id"] = transcripts["file_name"].str.removesuffix(".wav")
    for name, frame in (("transcripciones", transcripts), ("manifest", manifest)):
        if frame["anon_id"].isna().any() or frame["anon_id"].duplicated().any():
            raise ValueError(f"Identificadores vacíos o duplicados en {name}")
    if set(transcripts.anon_id) != set(manifest.anon_id):
        raise ValueError("Las transcripciones deben cubrir exactamente los IDs del manifest")
    if not transcripts.status.isin(["success", "empty"]).all():
        raise ValueError("Hay errores de transcripción pendientes; reinténtalos antes de extraer")
    transcripts["caller_transcript"] = transcripts["caller_transcript"].fillna("")
    has_text = transcripts.caller_transcript.str.strip().ne("")
    if not has_text.eq(transcripts.status.eq("success")).all():
        raise ValueError("El estado de transcripción no corresponde al texto")
    duration = pd.to_numeric(transcripts.duration_sec, errors="raise")
    if not (np.isfinite(duration) & duration.gt(0)).all():
        raise ValueError("La duración del audio debe ser finita y positiva")
    transcripts["duration_sec"] = duration
    merged = manifest[["anon_id", "label", "split"]].merge(
        transcripts[["anon_id", "caller_transcript", "status", "duration_sec"]],
        on="anon_id", validate="one_to_one")
    records = []
    for row in merged.itertuples():
        features = extract_language_features(row.caller_transcript, row.duration_sec)
        records.append({"anon_id": row.anon_id, "label": row.label, "split": row.split, **features})
    result = pd.DataFrame(records)
    if not np.isfinite(result[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Se generaron características no finitas")
    return result, merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.transcripts or args.root / "src/language/caller_transcriptions_corrected_sample.csv"
    out = args.output_dir or args.root / "review/language"
    features, calls = build_features(pd.read_csv(source), pd.read_csv(args.root / "manifest.csv"))
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "language_features_corrected.csv", index=False)
    # Cola para escucha humana: tres textos más cortos de cada clase en train.
    queue = calls.merge(features[["anon_id", "word_count"]], on="anon_id", validate="one_to_one")
    queue = queue[queue.split.eq("train")].sort_values(["word_count", "anon_id"])
    queue = queue.groupby("label", sort=True).head(3).copy()
    queue["audio_path"] = queue.anon_id.map(lambda x: str((args.root / "audio" / f"{x}.wav").resolve()))
    queue["review_status"] = "pendiente_escucha"
    queue["review_notes"] = ""
    review_path = out / "transcription_review.csv"
    # Las anotaciones humanas se conservan al repetir el comando.
    if not review_path.exists():
        queue.to_csv(review_path, index=False)
    print(f"Generadas {len(features)} llamadas, {len(FEATURES)} características: {out}")


if __name__ == "__main__":
    main()
