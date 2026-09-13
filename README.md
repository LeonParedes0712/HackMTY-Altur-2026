# Altur Challenge: HackMTY 2026

Recorded phone calls between a caller and a bank's AI customer-service agent, in Mexican Spanish.
In some calls the caller is a real person. In others the caller is an autonomous AI: speech recognition, a language model and a synthetic voice, dialing the same number.

Your task: given a call, decide whether the caller is human or synthetic.

## Files

| Path | Contents |
| --- | --- |
| `manifest.csv` | One row per call: `anon_id`, `label` (`human` or `synthetic`), `split` (`train` or `val`), `duration_s`. |
| `audio/<anon_id>.wav` | Stereo, 8 kHz, 16-bit PCM. Channel 0 is the caller (the one you classify). Channel 1 is the agent. |
| `turns/<anon_id>.json` | Speech segments per channel, `{"turns": [{"channel": 0, "start": 12.4, "end": 15.1}, ...]}`, seconds from the start of the file. Derived automatically from the audio; use them as a starting point. |

Audio is distributed as `altur-challenge-audio.zip` (see Releases). Unzip it in the repo root so the files land in `audio/`.

## The conversation

Every call follows the same customer-service flow, whoever is calling. The agent asks callers to repeat information back,
sometimes asks about things that do not exist, and there are moments where it interrupts, falls silent, or talks over the caller.
Both sides are given to you for a reason: channel 1 tells you what the caller was reacting to.

## Splits

`train` and `val` are speaker-disjoint: no caller appears in both. Judging uses a hidden set of calls from callers and voices that appear in neither split.

## Evaluation

Your system exposes `POST /detect`. It receives a stereo WAV clip (8 kHz, base64-encoded, channel 0 = caller, channel 1 = agent) and returns:

```json
{"is_synthetic": true, "confidence": 0.87}
```

`is_synthetic` is required. `confidence` is optional and used to break ties and reward calibration.

## Terms

Human callers volunteered, were told the call was recorded for an AI test, and used invented personal data. Do not try to identify anyone.
This dataset is provided for HackMTY 2026 only; do not redistribute.

## Detector final: instalación y demo

La decisión está congelada: FastAPI sirve exclusivamente el modelo acústico,
31 características del canal 0 y umbral **0.5**. No usa Whisper, transcripciones
externas ni JSON de turnos. No se requiere entrenar para ejecutar la entrega.

Desde la raíz de una clonación que incluya los archivos de entrega, con
**Python 3.14** (entorno verificado: Linux, Python 3.14.4):

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`requirements.txt` contiene solo el servidor y la inferencia acústica.
No necesita FFmpeg, CUDA ni pesos de Whisper. Para pruebas se usa
`requirements-test.txt`; `requirements-research.txt` es opcional para las
herramientas offline y no forma parte de la instalación del detector.

### Modelo incluido en la entrega

Archivo: **`models/acoustic_logistic.pkl`**, 4,329 bytes. La excepción específica
en `.gitignore` permite incluir este artefacto en el commit, conservando los
modelos experimentales ignorados. Debe acompañar al código: antes de ese
commit, una clonación del remoto todavía no contiene la entrega local.

SHA-256 del artefacto congelado:
`2a2b88f1cce06da5eba5364c6238e0bf27c83923f91ef9f79ba32f6cb8fcfd1e`.

Contiene imputación, escalado y regresión logística ya ajustados, orden de
variables y mapa de clases. No contiene audios ni transcripciones. Se carga
una vez por proceso al iniciar FastAPI; usar un worker mantiene una instancia.
Los audios del reto se obtienen por separado para la demo, conforme a sus
condiciones de uso; no se incluyen en el commit. No hay que regenerar el modelo.

### Endpoints y ejemplo

- `GET /health`: `{"status":"ok","model":"acoustic_logistic"}` cuando el modelo está cargado.
- `POST /detect`: JSON `{"audio":"<WAV completo en base64>"}`.
- `POST /detect-upload`: campo multipart `file`, accesible en Swagger: http://localhost:8000/docs.

Ambos POST llaman a la misma función `predict_audio` y al mismo predictor
acústico cargado al inicio. `/health` comprueba esa instancia sin ejecutar
una inferencia. La respuesta oficial de ambos POST es exactamente:

```json
{"is_synthetic": true}
```

`confidence` se omite: el contrato original lo declara opcional, pero no
define si representa probabilidad de synthetic o de la clase elegida.

Con un audio disponible localmente:

```bash
curl http://127.0.0.1:8000/health
.venv/bin/python -m src.acoustic.request_detection audio/call_04d682ac0cef.wav
curl -X POST http://127.0.0.1:8000/detect-upload -F 'file=@audio/call_04d682ac0cef.wav'
```

El comando `request_detection` construye el JSON base64 y lo envía a `/detect`.
Ejemplo equivalente explícito:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

body = json.dumps({
    "audio": base64.b64encode(Path("audio/call_04d682ac0cef.wav").read_bytes()).decode("ascii")
}).encode()
request = Request("http://127.0.0.1:8000/detect", data=body,
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=600) as response:
    print(json.load(response))
```

Se valida base64 estricto y WAV PCM de 16 bits, estéreo, 8000 Hz, duración
mayor que cero y hasta 600 segundos. Límite HTTP: 32 MiB, incluyendo base64
(o multipart); archivo decodificado: hasta 30 MiB. Se rechazan WAV truncados
y silencio total del canal del cliente. Son límites operativos, no límites
publicados por el reto. No se convierten formatos ni se mezclan canales.
Errores: 400 (base64/duración/silencio), 415 (formato), 413 (tamaño), 422
(estructura). Los archivos temporales de inferencia se limpian al terminar.

### Métricas finales y límites

No se reentrenó, calibró ni cambió el umbral para preparar esta entrega.
Resultados de la revisión anterior, mantenidos como evidencia:

| Evaluación | Modelo | Accuracy | F1 synthetic | ROC AUC | Log loss |
| --- | --- | ---: | ---: | ---: | ---: |
| CV interna de 5 particiones, 282 llamadas de train | Acústico reentrenado dentro de cada partición | 98.58% | 0.9882 | 0.9986 | 0.0502 |
| Misma CV interna | Lingüístico corregido | **78.72%** | 0.8276 | **0.8672** | 0.4573 |
| Val oficial, 71 llamadas; exploratorio | Artefacto acústico servido | 97.18% | 0.9714 | 1.0000 | 0.1074 |

El lingüístico corregido no se desplegó porque el acústico fue más estable
en la comparación interna (desviación de accuracy entre particiones: 0.71
frente a 3.08 puntos porcentuales), además de más preciso y operativo desde
el WAV recibido. Esto no prueba superioridad en hablantes nuevos. El lenguaje
requiere ASR interno integrado y medido; el conductual todavía requiere
turnos externos. No se despliegan fusiones.

La validación interna ajustó imputación, escalado y modelos exclusivamente
dentro de cada partición de train. No se encontraron duplicados exactos por
hash en los 353 audios. Sin embargo, no hay IDs de hablante/voz para agrupar
train: una partición aleatoria por llamada **no garantiza separación por
hablante**. Los hashes no descartan duplicados aproximados. Las métricas de
val ya se observaron y son **exploratorias**, no un test independiente; se
calcularon con características acústicas guardadas, no reprocesando todo val
por HTTP. No se afirma ausencia de sobreajuste ni desempeño garantizado en
voces ocultas. La calidad de ASR tampoco cuenta con una medición de WER.

### Pruebas y entrega

```bash
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -p 'test_api.py' -v
.venv/bin/python tests/smoke_uvicorn.py
```

Las pruebas reales requieren los audios del reto en `audio/`; usan la primera
llamada human y synthetic de train, sin elegirlas según predicciones. La
prueba HTTP inicia y detiene Uvicorn en localhost:8766 y comprueba health,
Swagger, ambos POST y entradas inválidas. No entrena ni transcribe.

Documentación: [operación FastAPI](docs/fastapi.md),
[revisión del modelo](docs/corrected-model-review.md) y
[archivos y verificación de entrega](docs/delivery.md).
