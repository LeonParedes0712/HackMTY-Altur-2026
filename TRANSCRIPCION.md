# Corrección de transcripción del caller

Se revisaron Git, `transcribe.py`, `language_features.py`, `manifest.csv` completo y `.gitignore` antes de editar. `src/language/train_model.py` no existe en esta copia. Los cambios acústicos locales preexistentes se conservaron; no se modificaron modelos ni características de lenguaje.

## Cambios

- Selecciona canal 0 antes de convertir y remuestrear.
- Convierte PCM entero o audio flotante a un arreglo mono `float32`. Los flotantes no se vuelven a dividir por 32768.
- Usa `scipy.signal.resample_poly`, ya disponible, con la frecuencia real del WAV y destino 16000 Hz. Mantiene la duración; no se limita a cambiar la etiqueta del sample rate.
- Carga `WhisperModel` una vez por lote y consume cada transcripción antes de continuar.
- Conserva `language="es"`, `beam_size=1`, CPU/int8 y modelo predeterminado `tiny`.
- Registra `success` (texto), `empty` (procesado sin texto) o `error` (incluye tipo y mensaje). También captura errores que aparecen al consumir el generador.
- Guarda un checkpoint atómico después de cada llamada. Una interrupción puede requerir repetir la llamada en curso, pero conserva las anteriores.
- Reanuda automáticamente; no vuelve a cargar Whisper cuando no hay pendientes. Por defecto tampoco repite errores: `--retry-errors` permite reintentarlos.
- Rechaza checkpoints incompatibles, modelos diferentes y audios modificados. Un bloqueo de archivo impide dos lotes simultáneos escribiendo la misma salida (macOS/Linux).
- Protege el CSV original y usa por defecto `caller_transcriptions_corrected.csv`.

## Uso desde la raíz del repositorio

En esta computadora, `python3` tiene faster-whisper instalado; `.venv/bin/python` no. No se cambió ningún entorno. La prueba usó faster-whisper 1.2.1, CTranslate2 4.7.1, SciPy 1.17.1, NumPy 2.3.4 y el modelo `small` ya descargado.

Muestra reproducida (dos llamadas antes vacías y dos con texto; las más cortas de cada grupo):

```sh
python3 src/language/transcribe.py --model small --sample-balanced --limit 4 --output src/language/caller_transcriptions_corrected_sample.csv
```

Repetir el comando reanuda sin volver a procesar esas cuatro llamadas. `--limit` limita la selección total, incluidas las ya procesadas; no significa N llamadas adicionales.

Para ampliar a todo el manifiesto conservando la muestra y el mismo modelo:

```sh
python3 src/language/transcribe.py --model small --output src/language/caller_transcriptions_corrected_sample.csv
```

O generar una versión completa separada con el modelo predeterminado `tiny` (puede necesitar descargarlo):

```sh
python3 src/language/transcribe.py
```

Cada CSV corregido tiene al lado un `.comparison.csv` por llamada y un `.summary.json` agregado. La comparación corresponde a todas las filas del checkpoint. Los errores anteriores son desconocidos: el programa anterior guardaba tanto los errores como las transcripciones sin texto como cadenas vacías.

## Resultados de la muestra real

| Métrica | CSV anterior, mismas 4 llamadas | Versión corregida con small |
|---|---:|---:|
| Textos vacíos | 2 | 0 |
| Errores registrados | No disponibles | 0 |
| Palabras totales | 2 | 199 |
| Procesadas con texto | 2 | 4 |

El CSV antiguo no registra el modelo realmente usado para generarlo. El código antiguo solicita `tiny`; esta comparación histórica por sí sola no permite atribuir toda la diferencia al remuestreo. Se ejecutó además una comparación controlada con `small`, idénticas opciones y los mismos cuatro WAV, cambiando únicamente el remuestreo. En esa comparación, los vacíos bajaron de 2 a 0 y las palabras subieron de 6 a 199, sin errores en ninguna variante. El resultado detallado está en `review/transcription/controlled_resampling.csv`.

Pruebas automatizadas:

```sh
python3 -m unittest discover -s tests -p test_transcribe.py -v
```

Nueve pruebas verifican conservación de duración y frecuencia de voz a 8/16/44.1 kHz, selección del caller, normalización PCM/flotante, los tres estados, errores del generador, continuidad del lote, carga única del modelo, reanudación, recuperación tras interrupción y protección del CSV original.

Para repetir la comparación controlada (usar una salida nueva):

```sh
PYTHONPATH=. python3 tests/compare_transcription_resampling.py --root . --sample src/language/caller_transcriptions_corrected_sample.csv --output review/transcription/controlled_resampling_repeat.csv --model small
```

## Pendiente

- Procesar las otras 349 llamadas después de revisar la muestra.
- Escuchar y corregir una referencia humana para medir exactitud de transcripción. Más palabras o menos vacíos no equivalen a menor tasa de error; hay frases y nombres que aún pueden estar mal transcritos.
- Verificar el desempeño con `tiny` si ese será el modelo final.
- `language_features.py` sigue leyendo el CSV original y apunta a `src/audio/audio_metrics.csv`, ausente. No se modificó esa integración ni se reentrenó ningún modelo.

No hubo commit, merge ni push. Los resultados de revisión se guardan fuera de `outputs/`, porque esa carpeta está ignorada por Git; así quedan visibles para revisión.
