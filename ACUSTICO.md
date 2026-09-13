# Detector acústico

La versión actual usa exclusivamente el canal 0 (cliente), extrae 31 características y aplica el modelo acústico existente. El umbral se conserva en 0.5. No requiere transcripciones ni los archivos de métricas de lenguaje.

## Ejecutar desde la raíz del proyecto

Con el entorno existente:

```sh
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

En otra terminal, enviar un audio:

```sh
.venv/bin/python -m src.acoustic.request_detection audio/call_04d682ac0cef.wav
```

También se puede predecir sin levantar el servicio:

```sh
.venv/bin/python -m src.acoustic.predict audio/call_04d682ac0cef.wav
```

La predicción directa muestra `p_synthetic`, `p_human` y el umbral. Estas probabilidades son estimaciones del modelo, no garantías.

## Contrato del servicio

- `GET /health`: devuelve `{"status": "ok", "model": "acoustic_logistic"}` después de cargar el modelo.
- `POST /detect`, `Content-Type: application/json`.
- Cuerpo: `{"audio": "BASE64_DEL_ARCHIVO_WAV_COMPLETO"}`.
- Respuesta: `{"is_synthetic": true, "confidence": 0.87}` (ejemplo).
- `confidence`: probabilidad de la clase elegida, finita entre 0 y 1; idéntica en JSON y upload.
- Errores: HTTP 400 (base64/duración/silencio), 415 (formato de audio), 413 (tamaño), 422 (estructura).
- WAV PCM de 16 bits, estéreo, 8000 Hz, hasta 600 segundos. Canal 0 = cliente; canal 1 = agente. Se rechaza el canal del cliente completamente silencioso.

El README del reto no especifica el nombre del campo de entrada. Aquí se usa `audio`; confirmar ese nombre con el evaluador antes de entregar. Por la especificación final de entrega, `confidence` usa `p_synthetic` si la clase elegida es synthetic y `p_human` en otro caso; una probabilidad no finita o fuera de [0, 1] produce HTTP 500.

Por defecto escucha únicamente en esta computadora, puerto 8000. Para un entorno de evaluación que necesite acceso desde fuera, usar `--host 0.0.0.0 --port 8000`. La inferencia se ejecuta en el pool de hilos de FastAPI; falta medir capacidad concurrente en despliegue.

## Evaluación reproducible

```sh
.venv/bin/python -m src.acoustic.evaluate
.venv/bin/python -m unittest discover -s tests -v
```

La evaluación carga el modelo guardado y las características existentes. No reentrena ni cambia el umbral. Genera:

- `outputs/acoustic_validation.json`: métricas y llamadas mal clasificadas.
- `outputs/acoustic_validation.csv`: predicciones de las 71 llamadas de validación.

Resultado comprobado: 69/71 correctas (97.18%), F1 sintético 0.9714. Matriz de confusión, filas reales y columnas predichas, orden human/synthetic: `[[35, 2], [0, 34]]`. Los dos errores son humanos clasificados como sintéticos. Esto no mide todavía el desempeño en el conjunto oculto.

## Preparar otro entorno

Se registraron las versiones del entorno local que pasó las pruebas, y se verificó la integración con Python 3.14:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

La instalación limpia del detector final se verificó con Python 3.14.4. El artefacto `models/acoustic_logistic.pkl` (4,329 bytes) está preparado para incluirse en el commit de entrega; los audios siguen excluidos. No se requiere reconstruirlo. Los comandos siguientes son únicamente referencia histórica de entrenamiento, no pasos de instalación de la entrega congelada:

```sh
.venv/bin/python src/acoustic/build_dataset.py
.venv/bin/python src/acoustic/train_model.py
```

Estos dos pasos regeneran las características y sobrescriben el modelo respectivamente; no se necesitan para usar el modelo actual.

## Nota histórica de evaluación

La rama original proponía investigar `call_10aa0d1d33f0` y `call_bc2c7c34acea`, los dos falsos positivos. Para nuevas decisiones se debe usar exclusivamente train; estos errores de val no deben orientar ajustes. Cualquier cambio en características debe aplicarse tanto al entrenamiento como a la predicción, reentrenarse y evaluarse. No ajustar características, modelos ni umbrales mirando val.

El servidor final es FastAPI. `src/acoustic/serve.py` se conserva solo como referencia histórica. Véase [documentación de FastAPI](docs/fastapi.md) para upload, pruebas y límites.
