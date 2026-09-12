"""Print metadata for up to five train calls per class; no identity analysis."""
import argparse
import csv
import wave
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    with (a.data / 'manifest.csv').open() as f:
        rows = list(csv.DictReader(f))
    for label in ('human', 'synthetic'):
        for row in [r for r in rows if r['label'] == label and r['split'] == 'train'][:5]:
            path = a.data / 'audio' / (row['anon_id'] + '.wav')
            with wave.open(str(path), 'rb') as w:
                print(row['anon_id'], label, {'sample_rate': w.getframerate(), 'channels': w.getnchannels(),
                      'sample_bits': w.getsampwidth() * 8, 'duration_s': w.getnframes() / w.getframerate(),
                      'compression': w.getcomptype()})

if __name__ == '__main__':
    main()
