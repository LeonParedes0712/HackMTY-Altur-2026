import json
import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np
from src.acoustic.preprocess import read_caller, caller_segments
from src.acoustic.features import extract_features, feature_names

class AcousticTests(unittest.TestCase):
    def test_channel_zero_and_pcm_scale(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.wav'
            data = np.column_stack([np.full(800, 1000), np.full(800, 20000)]).astype('<i2')
            with wave.open(str(p), 'wb') as w:
                w.setparams((2, 2, 8000, 0, 'NONE', 'not compressed'))
                w.writeframes(data.tobytes())
            audio, rate = read_caller(p)
            self.assertEqual(rate, 8000)
            np.testing.assert_allclose(audio, 1000 / 32768)

    def test_turns_exclude_agent_merge_and_clamp(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.json'
            p.write_text(json.dumps({'turns': [
                {'channel': 1, 'start': 0, 'end': 1},
                {'channel': 0, 'start': -.1, 'end': .3},
                {'channel': 0, 'start': .2, 'end': .4},
                {'channel': 0, 'start': .9, 'end': 2}]}))
            segments = caller_segments(np.ones(8000), 8000, p, 'turns')
            self.assertEqual([len(s) for s in segments], [3200, 800])

    def test_sine_frequency_and_energy(self):
        x = .5 * np.sin(2 * np.pi * 1000 * np.arange(8000) / 8000)
        features = extract_features([x])
        self.assertEqual(set(features), set(feature_names()))
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertAlmostEqual(features['energy_mean'], .125, places=5)
        self.assertAlmostEqual(features['spectral_centroid_mean'], 1000, delta=10)

    def test_no_frames_across_segment_boundaries(self):
        f = extract_features([np.ones(100), -np.ones(100)])
        self.assertTrue(np.isnan(list(f.values())).all())

    def test_silence_has_no_speech(self):
        self.assertEqual(caller_segments(np.zeros(8000), 8000), [])

if __name__ == '__main__':
    unittest.main()
