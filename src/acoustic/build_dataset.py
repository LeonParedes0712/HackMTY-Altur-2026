from pathlib import Path
import pandas as pd

from features import extract_features

ROOT = Path(__file__).resolve().parents[2]

df = pd.read_csv(ROOT / "manifest.csv")

test_df = pd.concat([
    df[df["label"] == "human"].head(5),
    df[df["label"] == "synthetic"].head(5)
])

for _, row in test_df.iterrows():
    anon_id = row["anon_id"]
    label = row["label"]

    audio_path = ROOT / "audio" / f"{anon_id}.wav"
    features = extract_features(audio_path)

    print(label, audio_path)