import base64
import csv
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient
from src.api import main

ROOT = Path(__file__).resolve().parents[1]


def wav_bytes(channels=2, rate=8000, width=2, frames=2048):
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as wav:
        wav.setparams((channels, width, rate, frames, 'NONE', 'not compressed'))
        wav.writeframes(b'\0' * frames * channels * width)
    return stream.getvalue()


class APIValidationTests(unittest.TestCase):
    def test_validation_and_lifecycle(self):
        with patch.object(main, 'AcousticPredictor') as factory:
            factory.return_value.predict.return_value = {'is_synthetic': True}
            with TestClient(main.app) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                self.assertEqual(client.get('/docs').status_code, 200)
                for kwargs in ({}, {'channels': 1}, {'rate': 16000}, {'width': 1}, {'frames': 0}):
                    data = wav_bytes(**kwargs)
                    expected = 200 if not kwargs else (400 if 'frames' in kwargs else 415)
                    for response in (
                        client.post('/detect', json={'audio': base64.b64encode(data).decode()}),
                        client.post('/detect-upload', files={'file': ('test.wav', data, 'audio/wav')}),
                    ):
                        self.assertEqual(response.status_code, expected, response.text)
                for data, expected in ((b'not wav', 415), (wav_bytes()[:-10], 400)):
                    self.assertEqual(client.post('/detect', json={'audio': base64.b64encode(data).decode()}).status_code, expected)
                for value in ('@@@', 'ááá', 'Y Q=='):
                    self.assertEqual(client.post('/detect', json={'audio': value}).status_code, 400)
                for payload in ({}, {'audio': 123}, {'audio': {'secret': 'private-audio'}}):
                    response = client.post('/detect', json=payload)
                    self.assertEqual(response.status_code, 422)
                    self.assertNotIn('private-audio', response.text)
                with patch.object(main, 'MAX_DURATION_S', 0.1):
                    self.assertEqual(client.post('/detect', json={'audio': base64.b64encode(wav_bytes()).decode()}).status_code, 400)
                with patch.object(main, 'MAX_FILE_SIZE', 10):
                    self.assertEqual(client.post('/detect-upload', files={'file': ('x.wav', wav_bytes())}).status_code, 413)
                with patch.object(main, 'MAX_BASE64_SIZE', 10):
                    self.assertEqual(client.post('/detect', json={'audio': 'A' * 12}).status_code, 413)
                with patch.object(main, 'MAX_REQUEST_SIZE', 10):
                    self.assertEqual(client.post('/detect', content=b'x' * 11).status_code, 413)
            factory.assert_called_once()
            self.assertIsNone(main.app.state.predictor)

    def test_silent_caller_is_client_error(self):
        from src.acoustic.preprocess import load_audio
        with patch.object(main, 'AcousticPredictor') as factory:
            factory.return_value.predict.side_effect = lambda path, **kwargs: load_audio(path)
            with TestClient(main.app) as client:
                data = wav_bytes()
                self.assertEqual(client.post('/detect', json={'audio': base64.b64encode(data).decode()}).status_code, 400)
                self.assertEqual(client.post('/detect-upload', files={'file': ('silent.wav', data)}).status_code, 400)

    def test_real_train_calls(self):
        with (ROOT / 'manifest.csv').open() as source:
            rows = list(csv.DictReader(source))
        # Primera llamada de cada clase en train, sin seleccionar por predicción.
        with TestClient(main.app) as client:
            for label in ('human', 'synthetic'):
                row = next(row for row in rows if row['split'] == 'train' and row['label'] == label)
                data = (ROOT / 'audio' / (row['anon_id'] + '.wav')).read_bytes()
                first = client.post('/detect', json={'audio': base64.b64encode(data).decode()})
                second = client.post('/detect-upload', files={'file': ('call.wav', data, 'audio/wav')})
                self.assertEqual(first.status_code, 200, first.text)
                self.assertEqual(second.status_code, 200, second.text)
                self.assertEqual(first.json(), second.json())
                self.assertEqual(set(first.json()), {'is_synthetic'})
                self.assertIs(type(first.json()['is_synthetic']), bool)
                print({'label': label, 'response': first.json()}, flush=True)


if __name__ == '__main__':
    unittest.main()
