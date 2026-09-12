import pandas as pd
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

manifest_path = ROOT / "manifest.csv"
audio_dir = ROOT / "audio"

df = pd.read_csv(manifest_path)

print("Primeras filas del manifest:")
print(df.head())

samples = pd.concat([
    df[df["label"] == "human"].head(5),
    df[df["label"] == "synthetic"].head(5)
])

print("\nRevisando audios:\n")

for _, row in samples.iterrows():
    anon_id = row["anon_id"]
    audio_path = audio_dir / f"{anon_id}.wav"

    audio, sr = sf.read(audio_path)

    print(
        anon_id,
        "| label:", row["label"],
        "| sr:", sr,
        "| shape:", audio.shape,
        "| duration:", round(len(audio) / sr, 2), "s"
    )