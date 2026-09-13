"""Servidor oficial: .venv/bin/python -m uvicorn src.api.main:app."""
import base64
import binascii
from contextlib import asynccontextmanager
import io
import math
from pathlib import Path
import tempfile
import wave

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictStr
from starlette.concurrency import run_in_threadpool

from src.acoustic.predict import AcousticPredictor
from src.acoustic.validation import MAX_BODY_BYTES, MAX_DURATION_SECONDS, validate_audio

MAX_FILE_SIZE = 30 * 1024 * 1024
MAX_BASE64_SIZE = 4 * ((MAX_FILE_SIZE + 2) // 3)
MAX_REQUEST_SIZE = MAX_BODY_BYTES
# Límites operativos; el README no especifica duración mínima/máxima.
MIN_DURATION_S = 1 / 8000
MAX_DURATION_S = MAX_DURATION_SECONDS


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = AcousticPredictor()
    yield
    app.state.predictor = None


app = FastAPI(title="Human Voice Detection API", version="1.0.0", lifespan=lifespan)


class BodyLimit:
    """Limita también cuerpos chunked antes del parseo JSON/multipart."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > MAX_REQUEST_SIZE:
                response = JSONResponse(status_code=413, content={"detail": "Solicitud demasiado grande"})
                return await response(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)

        async def replay():
            nonlocal body
            if body is not None:
                result, body = body, None
                return {"type": "http.request", "body": result, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app.add_middleware(BodyLimit)


class DetectRequest(BaseModel):
    audio: StrictStr


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # No reflejar el audio ni otros datos del cuerpo en errores de validación.
    return JSONResponse(status_code=422, content={"detail": "Se requiere JSON con audio base64 o un archivo multipart válido"})


def decode_audio(encoded: str) -> bytes:
    if len(encoded) > MAX_BASE64_SIZE:
        raise HTTPException(413, "El audio supera 30 MiB")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(400, "Base64 inválido") from exc


def validate_wav(data: bytes):
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "El audio supera 30 MiB")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
                raise HTTPException(415, "Se requiere WAV PCM de 16 bits")
            if wav.getframerate() != 8000 or wav.getnchannels() != 2:
                raise HTTPException(415, "Se requieren 8000 Hz y dos canales")
            frames = wav.getnframes()
            if not MIN_DURATION_S <= frames / 8000 <= MAX_DURATION_S:
                raise HTTPException(400, "Duración de audio inválida")
            if len(wav.readframes(frames)) != frames * 4:
                raise HTTPException(400, "Audio WAV truncado")
    except (wave.Error, EOFError) as exc:
        raise HTTPException(415, "Se requiere audio WAV PCM válido") from exc


def predict_audio(data: bytes, predictor: AcousticPredictor) -> DetectResponse:
    validate_wav(data)
    try:
        validate_audio(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    with tempfile.TemporaryDirectory(prefix="voice-detect-") as directory:
        path = Path(directory) / "audio.wav"
        path.write_bytes(data)
        try:
            result = predictor.predict(path, threshold=0.5)
        except ValueError as exc:
            raise HTTPException(400, "El audio no puede analizarse; verifica el canal del cliente") from exc
    is_synthetic = result["is_synthetic"]
    confidence = float(result["p_synthetic"] if is_synthetic else result["p_human"])
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise HTTPException(500, "El modelo devolvió una confianza inválida")
    return DetectResponse(is_synthetic=is_synthetic, confidence=confidence)


@app.get("/health")
def health(request: Request):
    if getattr(request.app.state, "predictor", None) is None:
        raise HTTPException(503, "Modelo no disponible")
    return {"status": "ok", "model": "acoustic_logistic"}


@app.post("/detect", response_model=DetectResponse)
async def detect(payload: DetectRequest, request: Request):
    data = decode_audio(payload.audio)
    return await run_in_threadpool(predict_audio, data, request.app.state.predictor)


@app.post("/detect-upload", response_model=DetectResponse)
async def detect_upload(request: Request, file: UploadFile = File(...)):
    try:
        data = await file.read(MAX_FILE_SIZE + 1)
        return await run_in_threadpool(predict_audio, data, request.app.state.predictor)
    finally:
        await file.close()
