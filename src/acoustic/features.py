import numpy as np
import librosa

from preprocess import load_audio


def extract_features(path):
    audio, sr = load_audio(path)

    # 13 MFCC: media y desviación estándar = 26 características
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
    )

    features = {}

    for index in range(13):
        number = index + 1
        features[f"mfcc_{number:02d}_mean"] = float(np.mean(mfcc[index]))
        features[f"mfcc_{number:02d}_std"] = float(np.std(mfcc[index]))

    # RMS por ventanas: describe cómo cambia la intensidad.
    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512,
    )[0]

    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # Frecuencia fundamental: un único valor por ventana, no varios candidatos.
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=70,
            fmax=400,
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
    except (ValueError, np.linalg.LinAlgError):
        f0 = None

    if f0 is None or len(f0) == 0:
        features["pitch_mean"] = np.nan
        features["pitch_std"] = np.nan
        features["voiced_fraction"] = 0.0
    else:
        valid_pitch = f0[np.isfinite(f0)]

        features["voiced_fraction"] = float(
            len(valid_pitch) / len(f0)
        )

        if len(valid_pitch) == 0:
            features["pitch_mean"] = np.nan
            features["pitch_std"] = np.nan
        else:
            features["pitch_mean"] = float(np.mean(valid_pitch))
            features["pitch_std"] = float(np.std(valid_pitch))

    return features


if __name__ == "__main__":
    result = extract_features("audio/call_04d682ac0cef.wav")

    print("Cantidad de características:", len(result))
    for name, value in result.items():
        print(f"{name}: {value}")
