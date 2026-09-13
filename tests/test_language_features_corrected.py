import unittest

import pandas as pd

try:
    from src.language.build_features import build_features, FEATURES
except ModuleNotFoundError:
    from build_features import build_features, FEATURES


class LanguageFeatureTests(unittest.TestCase):
    def setUp(self):
        self.manifest = pd.DataFrame({"anon_id": ["a", "b"], "label": ["human", "synthetic"],
                                      "split": ["train", "val"], "duration_s": [99, 99]})
        self.transcripts = pd.DataFrame({"file_name": ["b.wav", "a.wav"],
            "caller_transcript": ["", "hola hola"], "status": ["empty", "success"],
            "duration_sec": [4, 2]})

    def test_id_join_and_actual_duration(self):
        features, _ = build_features(self.transcripts, self.manifest)
        self.assertEqual(features.anon_id.tolist(), ["a", "b"])
        self.assertEqual(features.word_count.tolist(), [2, 0])
        self.assertEqual(features.words_per_second.tolist(), [1, 0])
        self.assertEqual(features.loc[0, "repetition_count"], 1)
        self.assertEqual(len(FEATURES), 9)

    def test_missing_and_duplicate_ids_rejected(self):
        for data in [self.transcripts.iloc[:1], pd.concat([self.transcripts, self.transcripts])]:
            with self.assertRaises(ValueError):
                build_features(data, self.manifest)

    def test_error_not_interpreted_as_empty_speech(self):
        self.transcripts.loc[0, "status"] = "error"
        with self.assertRaises(ValueError):
            build_features(self.transcripts, self.manifest)

    def test_invalid_duration_rejected(self):
        for duration in [0, -1, float("nan"), float("inf")]:
            data = self.transcripts.copy()
            data["duration_sec"] = [duration, 2]
            with self.assertRaises(ValueError):
                build_features(data, self.manifest)

    def test_status_text_mismatch_rejected(self):
        self.transcripts.loc[0, "status"] = "success"
        with self.assertRaises(ValueError):
            build_features(self.transcripts, self.manifest)


if __name__ == "__main__":
    unittest.main()
