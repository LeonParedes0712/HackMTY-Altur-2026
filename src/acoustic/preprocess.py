from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]


def load_audio(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    # Canal 0, sin mezclarlo con el otro canal.
    audio = audio[:, 0]

    if audio.size == 0:
        raise ValueError(f"Audio vacío: {path.name}")

    if not np.isfinite(audio).all():
        raise ValueError(f"Audio con valores inválidos: {path.name}")

    return audio, sr


if __name__ == "__main__":
    audio, sr = load_audio("audio/call_04d682ac0cef.wav")

    print("Sample rate:", sr)
    print("Shape:", audio.shape)
    print("Dimensiones:", audio.ndim)
    print("Duración:", round(len(audio) / sr, 2), "s")