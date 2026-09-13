"""Prueba HTTP real de Uvicorn; requiere sockets locales."""
import base64
import csv
from pathlib import Path
import subprocess
import sys
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'src.api.main:app', '--host', '127.0.0.1', '--port', '8766'], cwd=ROOT)
try:
    with httpx.Client(base_url='http://127.0.0.1:8766', timeout=180) as client:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('Uvicorn terminó antes de arrancar')
            try:
                response = client.get('/health')
                if response.status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            time.sleep(.1)
        else:
            raise RuntimeError('Uvicorn no arrancó')
        assert response.json() == {'status': 'ok', 'model': 'acoustic_logistic'}
        assert client.get('/docs').status_code == 200
        assert client.post('/detect', json={'audio': '@@@'}).status_code == 400
        assert client.post('/detect', json={'audio': base64.b64encode(b'not wav').decode()}).status_code == 415
        with (ROOT / 'manifest.csv').open() as source:
            rows = list(csv.DictReader(source))
        for label in ('human', 'synthetic'):
            row = next(row for row in rows if row['split'] == 'train' and row['label'] == label)
            data = (ROOT / 'audio' / (row['anon_id'] + '.wav')).read_bytes()
            first = client.post('/detect', json={'audio': base64.b64encode(data).decode()})
            second = client.post('/detect-upload', files={'file': ('call.wav', data, 'audio/wav')})
            assert first.status_code == second.status_code == 200
            assert first.json() == second.json()
            assert set(first.json()) == {'is_synthetic'}
            assert type(first.json()['is_synthetic']) is bool
            assert first.json()['is_synthetic'] == (label == 'synthetic')
            print(label, first.json(), flush=True)
        print('Uvicorn HTTP smoke: OK', flush=True)
finally:
    process.terminate()
    process.wait(timeout=15)
