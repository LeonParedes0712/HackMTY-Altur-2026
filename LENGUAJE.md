# Características de lenguaje corregidas

Se conectó el CSV corregido de 353 llamadas mediante `src/language/build_features.py`. Este nuevo punto de entrada reutiliza las nueve características existentes; elimina el sufijo `.wav` de los IDs y obtiene la duración real del WAV guardada en la transcripción. Valida cobertura completa, IDs únicos, estados, duración y valores finitos. No depende del archivo ausente `src/audio/audio_metrics.csv`.

Desde la raíz del proyecto:

```sh
.venv/bin/python -m src.language.build_features
.venv/bin/python -m src.language.evaluate_features
```

Aquí sí funciona `.venv`: estos pasos no utilizan Whisper. El script antiguo `language_features.py` permanece como referencia; usar `build_features` para procesar las transcripciones corregidas.

Los resultados se guardan en `review/language/`:

- `language_features_corrected.csv`: 353 IDs, etiquetas, particiones y nueve características.
- `model_comparison.json`: comparación acústica/lenguaje/combinación.
- `validation_predictions.csv`: probabilidades y aciertos por llamada.
- `transcription_review.csv`: seis llamadas de entrenamiento, tres humanas y tres sintéticas, seleccionadas por texto corto para revisión dirigida. Incluye ruta de audio, texto y columnas para notas. Repetir el comando no sobrescribe las anotaciones de este archivo.

`words_per_second` representa palabras por segundo de llamada completa, incluyendo pausas y el tiempo del agente. No representa la velocidad del cliente mientras habla. Las muletillas y reparaciones son conteos de patrones en el texto de Whisper; Whisper puede omitirlas o escribirlas mal.

## Comparación obtenida

Se fijaron tres regresiones logísticas con C=1 y umbral 0.5. Cada una se entrenó en memoria sobre las 282 llamadas de train; imputación y escalado se ajustaron exclusivamente con train. Se evaluaron las mismas 71 llamadas de val. No se guardó ni sustituyó ningún modelo.

| Variante | Aciertos | Errores | F1 sintético | Log loss (menor es mejor) |
|---|---:|---:|---:|---:|
| Acústica | 69/71 (97.18%) | 2 | 0.9714 | 0.1074 |
| Lenguaje | 57/71 (80.28%) | 14 | 0.7879 | 0.4085 |
| Combinación | 69/71 (97.18%) | 2 | 0.9714 | 0.0957 |

Agregar lenguaje no aumentó los aciertos. La pérdida de las probabilidades bajó ligeramente en esta validación; por sí sola no demuestra mejor calibración ni generalización. Es una comparación exploratoria sobre un val que ya se había observado. El detector servido sigue usando el modelo acústico existente.

## Revisión y próximos pasos

Escuchar las seis llamadas de la cola y anotar si se omitieron frases, se inventaron palabras o fallaron números. La cola prioriza casos sospechosos y no es una muestra representativa para estimar precisión global. Todavía no se compararon manualmente los audios contra los textos ni se midió WER.

Antes de ampliar el modelo, conviene probar señales de interacción (latencia de respuesta, pausas y solapamientos) en un experimento separado y con evaluación definida de antemano. La combinación actual no justifica agregar el costo de Whisper al servicio.

Pruebas de integridad:

```sh
.venv/bin/python -m unittest discover -s tests -p test_language_features_corrected.py -v
```

Se conservaron transcripciones originales, características antiguas, modelos y cambios locales previos. No se hizo commit, merge ni push.
