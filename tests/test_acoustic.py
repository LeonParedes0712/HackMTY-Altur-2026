import base64
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from unittest.mock import Mock

import numpy as np
import soundfile as sf

from src.acoustic.preprocess import load_audio
from src.acoustic.serve import detect, make_handler


def wav(sr=8000, channels=2, silent=False, subtype="PCM_16"):
    audio = np.zeros((800, channels), dtype=np.float32)
    if not silent:
        audio[:, 0] = 0.1 * np.sin(np.arange(800) * 0.2)
    if channels == 2:
        audio[:, 1] = 0.7
    out = io.BytesIO()
    sf.write(out, audio, sr, format="WAV", subtype=subtype)
    return out.getvalue()


def payload(raw):
    return {"audio": base64.b64encode(raw).decode("ascii")}


class AcousticTests(unittest.TestCase):
    def test_caller_channel_only(self):
        audio, sr = load_audio(io.BytesIO(wav()))
        self.assertEqual(sr, 8000)
        self.assertEqual(audio.ndim, 1)
        self.assertLess(float(np.max(np.abs(audio))), 0.11)

    def test_wrong_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "8000"):
            load_audio(io.BytesIO(wav(sr=16000)))

    def test_mono_rejected(self):
        with self.assertRaisesRegex(ValueError, "estéreo"):
            load_audio(io.BytesIO(wav(channels=1)))

    def test_silent_caller_rejected_even_when_agent_speaks(self):
        with self.assertRaisesRegex(ValueError, "silencio"):
            load_audio(io.BytesIO(wav(silent=True)))

    def test_bad_requests_never_call_model(self):
        predictor = Mock()
        for body in [None, {}, {"audio": "%%%"}, {"audio": ""}, payload(b"not wav"),
                     payload(wav(channels=1)), payload(wav(sr=16000)), payload(wav(subtype="FLOAT"))]:
            with self.subTest(body=str(body)[:30]):
                with self.assertRaises(ValueError):
                    detect(body, predictor)
        predictor.predict.assert_not_called()

    def test_detect_contract(self):
        predictor = Mock()
        predictor.predict.return_value = {"is_synthetic": True}
        self.assertEqual(detect(payload(wav()), predictor), {"is_synthetic": True})
        self.assertEqual(predictor.predict.call_args.args[0].getvalue(), wav())

    def test_http_contract(self):
        predictor = Mock()
        predictor.predict.return_value = {"is_synthetic": False}
        server = HTTPServer(("127.0.0.1", 0), make_handler(predictor))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(url+"/health") as response:
                self.assertEqual(json.load(response), {"status": "ok"})
            request = urllib.request.Request(url+"/detect", data=json.dumps(payload(wav())).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request) as response:
                self.assertEqual(json.load(response), {"is_synthetic": False})
            request = urllib.request.Request(url+"/detect", data=b'{', headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
