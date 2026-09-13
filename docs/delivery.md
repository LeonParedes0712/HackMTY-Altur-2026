# Entrega congelada

Rama: `feature/behavior-leo`. HEAD: `a3bb91a`. No se hizo commit, push, merge,
entrenamiento, transcripción ni ajuste de umbral en esta preparación.
Se conservan los cambios locales y el stash de respaldo de la integración.

## Clasificación del estado local

### Detector final, compatibilidad y documentación de entrega

La lista exacta propuesta para el commit de entrega es:

```text
.gitignore
README.md
ACUSTICO.md
requirements.txt
requirements-test.txt
requirements-research.txt
models/acoustic_logistic.pkl
src/api/main.py
src/acoustic/features.py
src/acoustic/predict.py
src/acoustic/validation.py
src/acoustic/serve.py
docs/fastapi.md
docs/corrected-model-review.md
docs/delivery.md
tests/test_api.py
tests/smoke_uvicorn.py
```

Son 17 archivos. `src/acoustic/preprocess.py` y
`src/acoustic/request_detection.py` ya están registrados y no tienen cambios.
`serve.py` queda solo como referencia y soporte de tests históricos; su entrada
no inicia otro servidor. `requirements-research.txt` no es necesario para
inferencia, pero documenta la separación de las dependencias offline que antes
estaban mezcladas en requirements.txt. El resumen de revisión del modelo se
incluye como evidencia de la decisión; sus artefactos detallados son opcionales.

No se ejecutó `git add`: la lista es para revisión y commit posterior. El
artefacto todavía no está registrado, pero ya no está ignorado. Una clonación
del remoto actual no incluye estos cambios hasta que se incorporen allí.

### Resultados experimentales: pueden quedar fuera del commit de entrega

Se conservan localmente; no son entradas del detector:

```text
src/fusion/train_fusion.py
src/language/train_model.py
src/language/review_corrected_model.py
tests/test_corrected_model_review.py
review/corrected_model_20260913T101639050249Z/acoustic_fold_audit.json
review/corrected_model_20260913T101639050249Z/audio_hashes.csv
review/corrected_model_20260913T101639050249Z/duplicates.json
review/corrected_model_20260913T101639050249Z/language_features_verified.csv
review/corrected_model_20260913T101639050249Z/language_fold_audit.json
review/corrected_model_20260913T101639050249Z/metadata_inventory.json
review/corrected_model_20260913T101639050249Z/provenance.json
review/corrected_model_20260913T101639050249Z/report.json
review/corrected_model_20260913T101639050249Z/train_oof_predictions.csv
```

El directorio de revisión ocupa aproximadamente 248 KiB. Si se quiere versionar
la reproducción completa de la revisión, estos scripts/tests/resultados pueden
ir en un commit separado. El resumen documental de entrega no depende de
ellos para iniciar el servidor. Los modelos lingüísticos/fusiones en `models/`
y las tablas/modelos de `outputs/` permanecen ignorados y fuera de la entrega.
Los resultados ya registrados por la integración anterior no fueron retirados.

### Temporales o copias redundantes: conservar fuera de Git, sin borrar

- `.venv/`, `__pycache__/`, `*.pyc`: entorno y cachés locales ignorados.
- `review/.../language_features_verified.csv`: copia de auditoría del CSV ya
  existente en `review/language/`; clasificada arriba como evidencia opcional.
- `requirements-acoustic.txt`: lista anterior ya registrada, parcialmente
  redundante con requirements.txt; se conserva sin cambios como referencia.
- `/tmp/hackmty26-before-integration-20260913-041053/` y `stash@{0}`: respaldos
  de la integración; no se tocaron ni eliminaron.
- `/tmp/hackmty26-api-main-before.py`: respaldo anterior del servidor.
- `/tmp/hackmty26-delivery-clean/`: entorno de instalación limpia.
- `/tmp/hackmty26-delivery-source/`: copia aislada de los archivos de ejecución
  con dos WAV exclusivamente para prueba local; no forma parte del commit.
- `/tmp/hackmty26-delivery-numba/` y `/tmp/hackmty26-numba/`: cachés de prueba.

Los audios son datos locales necesarios para la demo, no temporales; no se
incluyen ni borran. El README del reto restringe la redistribución del dataset.

## Artefacto de producción

`/detect` y `/detect-upload` usan `predict_audio` y la misma instancia de
`AcousticPredictor`. `/health` verifica esa instancia sin realizar predicción.
Archivo: `models/acoustic_logistic.pkl`, **4,329 bytes**, 31 características,
Pipeline de imputación, escalado y regresión logística, umbral fijo **0.5**.

SHA-256 antes y después de preparar la entrega:
`2a2b88f1cce06da5eba5364c6238e0bf27c83923f91ef9f79ba32f6cb8fcfd1e`.

No estaba registrado; `.gitignore` ignoraba `models/`. Ahora permite únicamente
este archivo. Es pequeño: incluirlo en Git normal evita entrenamientos de
instalación y no requiere LFS. El artefacto inspeccionado contiene el modelo,
orden de variables y mapa de clases; no audios ni transcripciones. No se
identificó en el README una prohibición específica de entregar los parámetros
aprendidos del detector; se mantiene la restricción sobre los datos originales.

La respuesta de ambos POST es `{"is_synthetic": boolean, "confidence": float}`.
Por la especificación final, confidence es la probabilidad de la clase elegida
(`p_synthetic` si true, `p_human` si false), finita y entre 0 y 1. Un valor
inválido produce HTTP 500. No se recalibra ni se cambia el umbral 0.5.

## Verificación final

- Instalación real desde cero en `/tmp/hackmty26-delivery-clean/`, creado con
  `.venv/bin/python -m venv`, Python 3.14.4 Linux x86_64, sin paquetes del
  entorno original. `pip install -r requirements-test.txt` y `pip check`: OK.
- `requirements.txt` instala solo servidor/inferencia; las dependencias de
  test están separadas. Whisper, CUDA y FFmpeg no se necesitan para servir.
- Copia aislada sin outputs, transcripciones ni turnos: Uvicorn cargó el
  artefacto incluido y pasó HTTP real usando las dependencias recién instaladas.
- `/health`: 200 y modelo acoustic_logistic. Swagger `/docs`: 200.
- Primera llamada human de train: 200, `is_synthetic=false` por JSON y upload.
- Primera llamada synthetic de train: 200, `is_synthetic=true` por ambos POST.
- Base64 inválido: 400; contenido no WAV: 415. Respuestas positivas con la
  claves is_synthetic y confidence, tipos booleano/float, confianza finita
  entre 0 y 1 e igualdad entre endpoints.
- 24 tests pasaron en el entorno original (6.118 s): API, acústico, integridad
  de características lingüísticas y procesamiento de transcripción con mocks.
  No se ejecutó Whisper ni entrenamiento; no se ejecutó el test experimental
  que ajusta clasificadores sobre datos de juguete.
- Compilación de archivos de entrega y `git diff --check`: OK.
- TestClient emite un aviso de deprecación del adaptador httpx; funciona con
  las versiones fijadas. La prueba Uvicorn usa HTTP real y no ese adaptador.

Prueba HTTP reproducible, después de instalar `requirements-test.txt` y
colocar los audios del reto en audio/:

```bash
.venv/bin/python tests/smoke_uvicorn.py
```

Comandos de demo:

```bash
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
# En otra terminal:
curl http://127.0.0.1:8000/health
.venv/bin/python -m src.acoustic.request_detection audio/call_04d682ac0cef.wav
curl -X POST http://127.0.0.1:8000/detect-upload -F 'file=@audio/call_04d682ac0cef.wav'
```

Swagger: http://localhost:8000/docs. La primera inferencia puede tardar más
por compilación/caché de Numba. Falta dimensionar latencia y concurrencia en
el equipo final; las pruebas de contrato no equivalen a una prueba de carga.

Las métricas y límites de generalización están en README y en
`docs/corrected-model-review.md`; no se recalcularon durante esta preparación.

## Cambio final de confidence

Después de la entrega inicial se añadió confidence a ambos POST, sin cambiar
modelo ni umbral. La suite completa registrada pasó: 26 tests (5.435 s).
Se excluyen únicamente los experimentos sin registrar que entrenan modelos.
HTTP real con Uvicorn pasó para ambas clases y ambos POST con resultados iguales:

- Human: `{"is_synthetic":false,"confidence":0.9969666626337651}`.
- Synthetic: `{"is_synthetic":true,"confidence":0.9797105228621958}`.

También pasaron health, Swagger y rechazos de base64/formato inválidos. El hash
del modelo acústico sigue siendo el documentado. Archivos exactos para el commit
de este cambio (la lista de 17 archivos anterior corresponde a la entrega inicial):

```text
src/api/main.py
tests/test_api.py
tests/smoke_uvicorn.py
README.md
ACUSTICO.md
docs/fastapi.md
docs/delivery.md
```
