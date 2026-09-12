import numpy as np
import librosa

from preprocess import load_audio


def extract_energy(audio):
    energy = np.mean(audio ** 2)
    return energy


def extract_pitch(audio, sr):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

    pitch_values = pitches[(pitches > 70) & (pitches < 400)]

    if len(pitch_values) == 0:
        return 0

    return np.mean(pitch_values)


if __name__ == "__main__":
    audio, sr = load_audio("audio/call_04d682ac0cef.wav")

    energy = extract_energy(audio)
    pitch = extract_pitch(audio, sr)

    print("Energy:", energy)
    print("Pitch:", pitch)