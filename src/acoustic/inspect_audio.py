from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]


def main():
    df = pd.read_csv(
        ROOT / "manifest.csv",
        dtype={"anon_id": "string", "label": "string"},
    )

    required = {"anon_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")

    if df[["anon_id", "label"]].isna().any().any():
        raise ValueError("Hay filas sin anon_id o label")

    df["anon_id"] = df["anon_id"].str.strip()
    df["label"] = df["label"].str.strip().str.lower()

    if df["anon_id"].eq("").any():
        raise ValueError("Hay identificadores vacíos")

    if df["anon_id"].duplicated().any():
        raise ValueError("Hay anon_id duplicados")

    unexpected = set(df["label"]) - {"human", "synthetic"}
    if unexpected:
        raise ValueError(f"Etiquetas inesperadas: {sorted(unexpected)}")

    print("Cantidad de audios por clase:")
    print(df["label"].value_counts())

    errors = 0

    for label in ("human", "synthetic"):
        samples = df[df["label"] == label].head(5)

        if samples.empty:
            print(f"AVISO: no hay audios de la clase {label}")
            continue

        for _, row in samples.iterrows():
            anon_id = row["anon_id"]
            path = ROOT / "audio" / f"{anon_id}.wav"

            try:
                info = sf.info(str(path))

                if info.frames == 0:
                    raise ValueError("Audio vacío")

                print(
                    anon_id,
                    "| label:", label,
                    "| sr:", info.samplerate,
                    "| canales:", info.channels,
                    "| duración:", round(info.duration, 2), "s",
                )
            except (OSError, RuntimeError, ValueError) as exc:
                errors += 1
                print(f"ERROR en {anon_id}: {exc}")

    print(f"\nInspección terminada. Errores: {errors}")


if __name__ == "__main__":
    main()