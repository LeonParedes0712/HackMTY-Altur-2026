# Detector acústico

La versión actual usa exclusivamente el canal 0 (cliente), extrae 31 características y aplica el modelo acústico existente. El umbral se conserva en 0.5. No requiere transcripciones ni los archivos de métricas de lenguaje.

## Ejecutar desde la raíz del proyecto

Con el entorno existente:

```sh
.venv/bin/python -m src.acoustic.serve
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

- `GET /health`: devuelve `{"status": "ok"}` después de cargar el modelo.
- `POST /detect`, `Content-Type: application/json`.
- Cuerpo: `{"audio": "BASE64_DEL_ARCHIVO_WAV_COMPLETO"}`.
- Respuesta: `{"is_synthetic": true}` o `{"is_synthetic": false}`.
- Errores de entrada: HTTP 400; tipo de contenido incorrecto: 415; cuerpo vacío o mayor de 32 MiB: 413.
- WAV PCM de 16 bits, estéreo, 8000 Hz, hasta 600 segundos. Canal 0 = cliente; canal 1 = agente. Se rechaza el canal del cliente completamente silencioso.

El README del reto no especifica el nombre del campo de entrada. Aquí se usa `audio`; confirmar ese nombre con el evaluador antes de entregar. Se omite `confidence`, que es opcional, hasta confirmar si representa probabilidad de voz sintética o confianza de la clase elegida.

Por defecto escucha únicamente en esta computadora, puerto 8000. Para un entorno de evaluación que necesite acceso desde fuera, usar `--host 0.0.0.0 --port 8000`. Es un servicio de demostración que procesa una llamada a la vez; no está dimensionado para tráfico concurrente.

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

Se registraron las versiones del entorno local que pasó las pruebas, con Python 3.13:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-acoustic.txt
```

La instalación desde cero no se ha verificado. El modelo `models/acoustic_logistic.pkl` y los datos acústicos están excluidos de Git por el proyecto; no basta con copiar únicamente el código. Para reconstruir el modelo, con los audios y manifest disponibles:

```sh
.venv/bin/python src/acoustic/build_dataset.py
.venv/bin/python src/acoustic/train_model.py
```

Estos dos pasos regeneran las características y sobrescriben el modelo respectivamente; no se necesitan para usar el modelo actual.

## Siguiente mejora

Investigar `call_10aa0d1d33f0` y `call_bc2c7c34acea`, los dos falsos positivos. Después comparar extracción sobre segmentos de voz frente a la llamada completa. Cualquier cambio en características debe aplicarse tanto al entrenamiento como a la predicción, reentrenarse y evaluarse. No ajustar el umbral únicamente para corregir estos dos casos.
