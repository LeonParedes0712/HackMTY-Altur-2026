"""Read original PCM samples and isolate caller speech without joining cuts."""
import json
import wave
from pathlib import Path
import numpy as np


def read_caller(path):
    with wave.open(str(path), 'rb') as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        if (rate, channels, width, wav.getcomptype()) != (8000, 2, 2, 'NONE'):
            raise ValueError(f'{path}: expected stereo 8 kHz 16-bit PCM WAV')
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
    return samples.reshape(-1, 2)[:, 0].astype(np.float64) / 32768.0, rate


def energy_intervals(audio, rate):
    """Basic fallback VAD, not a trained speech detector. Same rule at inference."""
    hop = rate // 50
    bounds = [(i, min(i + hop, len(audio))) for i in range(0, len(audio), hop)]
    if not bounds:
        return []
    rms = np.array([np.sqrt(np.mean(audio[a:b] ** 2)) for a, b in bounds])
    threshold = max(0.003, min(float(np.percentile(rms, 20)) * 3,
                               float(np.percentile(rms, 90)) * 0.3))
    return [bounds[i] for i in np.flatnonzero(rms > threshold)]


def caller_segments(audio, rate, turns_path=None, mode='energy'):
    if mode == 'turns':
        if turns_path is None or not Path(turns_path).is_file():
            raise FileNotFoundError(f'Missing turns: {turns_path}')
        turns = json.loads(Path(turns_path).read_text())['turns']
        intervals = []
        for turn in turns:
            if turn['channel'] != 0:
                continue
            start, end = float(turn['start']), float(turn['end'])
            if not np.isfinite([start, end]).all() or end <= start:
                raise ValueError('Invalid caller turn')
            a, b = max(0, int(start * rate)), min(len(audio), int(end * rate))
            if b > a:
                intervals.append((a, b))
    elif mode == 'energy':
        intervals = energy_intervals(audio, rate)
    else:
        raise ValueError('mode must be energy or turns')
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return [audio[a:b] for a, b in merged]
