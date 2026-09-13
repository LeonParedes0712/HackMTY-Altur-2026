"""Servicio local del detector acústico: python -m src.acoustic.serve."""
import argparse
import base64
import binascii
import io
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

import soundfile as sf

try:
    from .predict import AcousticPredictor
except ImportError:
    from predict import AcousticPredictor

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_DURATION_SECONDS = 600


def detect(payload, predictor):
    if not isinstance(payload, dict) or not isinstance(payload.get("audio"), str):
        raise ValueError("Envía un objeto JSON con el campo audio en base64")
    try:
        raw = base64.b64decode(payload["audio"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("El campo audio no contiene base64 válido") from exc
    if not raw:
        raise ValueError("Audio vacío")
    try:
        info = sf.info(io.BytesIO(raw))
    except (RuntimeError, ValueError) as exc:
        raise ValueError("No se pudo leer el audio WAV") from exc
    if info.format != "WAV" or info.subtype != "PCM_16":
        raise ValueError("Se requiere WAV PCM de 16 bits")
    if info.samplerate != 8000 or info.channels != 2:
        raise ValueError("Se requiere WAV estéreo a 8000 Hz")
    if not 0 < info.duration <= MAX_DURATION_SECONDS:
        raise ValueError("La duración debe ser mayor que cero y como máximo 600 segundos")
    result = predictor.predict(io.BytesIO(raw))
    # confidence es opcional; se omite hasta confirmar su semántica con el reto.
    return {"is_synthetic": result["is_synthetic"]}


def make_handler(predictor):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def respond(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ok"})
            else:
                self.respond(404, {"error": "Ruta inexistente"})

        def do_POST(self):
            if self.path != "/detect":
                self.respond(404, {"error": "Ruta inexistente"})
                return
            if self.headers.get_content_type() != "application/json":
                self.respond(415, {"error": "Usa Content-Type: application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    self.respond(413, {"error": "El cuerpo debe contener entre 1 byte y 32 MiB"})
                    return
                payload = json.loads(self.rfile.read(length))
                self.respond(200, detect(payload, predictor))
            except (ValueError, UnicodeError, sf.LibsndfileError) as exc:
                self.respond(400, {"error": str(exc)})
            except TimeoutError:
                self.respond(408, {"error": "Se agotó el tiempo para recibir el audio"})
            except Exception:
                logging.exception("Fallo de inferencia acústica")
                self.respond(500, {"error": "Fallo interno del detector"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    predictor = AcousticPredictor()
    server = HTTPServer((args.host, args.port), make_handler(predictor))
    print(f"Detector listo en http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
