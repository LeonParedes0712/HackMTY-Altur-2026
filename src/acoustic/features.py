"""25 ms frames, 10 ms hop; aggregate frames, never concatenate speech cuts."""
import argparse
import csv
import wave
from pathlib import Path
import numpy as np
from .preprocess import read_caller, caller_segments


def feature_names():
    return ([f'mfcc_{i}_{stat}' for i in range(1, 14) for stat in ('mean', 'std')]
            + [f'energy_{s}' for s in ('mean', 'std', 'min', 'max')]
            + [f'{name}_{s}' for name in ('zcr', 'spectral_centroid', 'spectral_rolloff')
               for s in ('mean', 'std')])


def extract_features(segments, rate=8000):
    if rate != 8000:
        raise ValueError('Expected 8000 Hz')
    size, hop, nfft, bands = 200, 80, 256, 26
    blocks = [np.lib.stride_tricks.sliding_window_view(s, size)[::hop]
              for s in segments if len(s) >= size]
    names = feature_names()
    if not blocks:
        return dict.fromkeys(names, float('nan'))
    frames = np.concatenate(blocks)
    power = np.abs(np.fft.rfft(frames * np.hanning(size), n=nfft)) ** 2 / nfft
    freqs = np.fft.rfftfreq(nfft, 1 / rate)
    mel_max = 2595 * np.log10(1 + rate / 2 / 700)
    edges = 700 * (10 ** (np.linspace(0, mel_max, bands + 2) / 2595) - 1)
    bank = np.array([np.maximum(0, np.minimum((freqs - a) / (b - a),
                                             (c - freqs) / (c - b)))
                     for a, b, c in zip(edges, edges[1:], edges[2:])])
    logmel = np.log(np.maximum(power @ bank.T, 1e-12))
    # Orthonormal DCT-II, coefficients C0..C12, named mfcc_1..mfcc_13.
    dct = np.cos(np.pi / bands * (np.arange(bands) + 0.5)[None, :]
                 * np.arange(13)[:, None]) * np.sqrt(2 / bands)
    dct[0] /= np.sqrt(2)
    mfcc = logmel @ dct.T
    result = {}
    for i in range(13):
        result[f'mfcc_{i+1}_mean'] = float(mfcc[:, i].mean())
        result[f'mfcc_{i+1}_std'] = float(mfcc[:, i].std())
    energy = np.mean(frames ** 2, axis=1)
    for stat in ('mean', 'std', 'min', 'max'):
        result[f'energy_{stat}'] = float(getattr(energy, stat)())
    mass = power.sum(axis=1)
    centroid = (power @ freqs) / np.maximum(mass, 1e-12)
    rolloff = freqs[np.argmax(np.cumsum(power, axis=1) >= 0.85 * mass[:, None], axis=1)]
    zcr = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)
    for name, values in [('zcr', zcr), ('spectral_centroid', centroid), ('spectral_rolloff', rolloff)]:
        result[f'{name}_mean'] = float(values.mean())
        result[f'{name}_std'] = float(values.std())
    return result


def extract_call(path, turns_path=None, mode='energy'):
    audio, rate = read_caller(path)
    segments = caller_segments(audio, rate, turns_path, mode)
    values = extract_features(segments, rate)
    diagnostics = {'caller_duration_s': sum(map(len, segments)) / rate,
                   'status': 'ok' if np.isfinite(list(values.values())).all() else 'no_usable_speech'}
    return values, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/acoustic_features.csv'))
    parser.add_argument('--mode', choices=['energy', 'turns'], default='energy')
    args = parser.parse_args()
    with (args.data / 'manifest.csv').open() as f:
        rows = list(csv.DictReader(f))
    ids = [r['anon_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate anon_id in manifest')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['anon_id'] + feature_names() + ['label', 'split', 'caller_duration_s', 'status', 'segmentation']
    errors = []
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            aid = row['anon_id']
            if Path(aid).name != aid or row['label'] not in ('human', 'synthetic') or row['split'] not in ('train', 'val'):
                raise ValueError(f'Invalid manifest row: {aid}')
            try:
                values, diag = extract_call(args.data / 'audio' / f'{aid}.wav',
                                           args.data / 'turns' / f'{aid}.json', args.mode)
            except (OSError, ValueError, KeyError, EOFError, wave.Error) as exc:
                values = dict.fromkeys(feature_names(), float('nan'))
                diag = {'caller_duration_s': 0, 'status': 'error'}
                errors.append({'anon_id': aid, 'error': str(exc)})
            writer.writerow({'anon_id': aid, **values, 'label': row['label'], 'split': row['split'],
                             **diag, 'segmentation': args.mode})
    import json
    args.output.with_suffix('.errors.json').write_text(json.dumps(errors, indent=2))
    print(f'{len(rows)} rows; {len(errors)} errors; output: {args.output}')
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
