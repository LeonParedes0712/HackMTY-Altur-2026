"""Validación WAV compartida con la referencia acústica original."""
import io
import soundfile as sf

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_DURATION_SECONDS = 600


def validate_audio(raw):
    if not raw:
        raise ValueError("Audio vacío")
    try:
        info = sf.info(io.BytesIO(raw))
    except (RuntimeError, ValueError) as exc:
        raise ValueError("No se pudo leer el audio WAV") from exc
    if info.format != "WAV" or info.subtype != "PCM_16":
        raise ValueError("Se requiere WAV PCM de 16 bits")
    if info.samplerate != 8000 or info.channels != 2:
        raise ValueError("Se requiere WAV estéreo a 8000 Hz")
    if not 0 < info.duration <= MAX_DURATION_SECONDS:
        raise ValueError("La duración debe ser mayor que cero y como máximo 600 segundos")
