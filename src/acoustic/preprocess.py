from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]


def load_audio(path):
    # Acepta rutas y WAV en memoria para compartir la extracción con la API.
    if isinstance(path, (str, Path)):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path

    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    if sr != 8000:
        raise ValueError("Se requiere audio a 8000 Hz")
    if audio.shape[1] != 2:
        raise ValueError("Se requiere audio estéreo: canal 0 cliente, canal 1 agente")

    # Canal 0, sin mezclarlo con el otro canal.
    audio = audio[:, 0]

    if audio.size == 0:
        raise ValueError("Audio vacío")

    if not np.isfinite(audio).all():
        raise ValueError("Audio con valores inválidos")

    if not np.any(audio):
        raise ValueError("El canal del cliente contiene únicamente silencio")

    return audio, sr


if __name__ == "__main__":
    audio, sr = load_audio("audio/call_04d682ac0cef.wav")

    print("Sample rate:", sr)
    print("Shape:", audio.shape)
    print("Dimensiones:", audio.ndim)
    print("Duración:", round(len(audio) / sr, 2), "s")
