import json
from pathlib import Path

import numpy as np
import soundfile as sf

def load_audio(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio[:, 0]

    return audio, sr

if __name__ == "__main__":
    audio, sr = load_audio("audio/call_04d682ac0cef.wav")

    print("Sample rate:", sr)
    print("Shape:", audio.shape)
    print("Dimensiones:", audio.ndim)