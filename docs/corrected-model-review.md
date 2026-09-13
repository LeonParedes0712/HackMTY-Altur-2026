# Revisión del modelo con transcripciones corregidas

Se conserva el modelo acústico en `/detect` y `/detect-upload`:
`models/acoustic_logistic.pkl`, cargado por `AcousticPredictor` una vez al
arrancar FastAPI, con umbral 0.5. No se modificó el servidor ni ese artefacto.
La nueva logística lingüística es un modelo de evaluación offline, no un
reemplazo del predictor de producción.

## Datos y procedencia

Entrada: `src/language/caller_transcriptions_corrected_sample.csv`. Pese al
nombre `sample`, contiene exactamente los **353 IDs** del manifest, únicos,
con estado `success`: 282 en train (113 human, 169 synthetic) y 71 en val.
No se retranscribió ningún audio.

`review/language/language_features_corrected.csv` ya existía. Se reconstruyeron
sus características en memoria usando `build_features` y se comprobó igualdad
numérica y de metadatos, además de cobertura exacta. No fue necesario regenerar
ni sobrescribir el CSV. Se guardó una copia verificada en el nuevo directorio
de revisión. Las transcripciones originales, resultados previos y modelos
existentes se preservaron; se verificaron por SHA-256 30 archivos preexistentes.

La fuente corregida registra Whisper `small` y remuestreo del canal 0 de
8000 a 16000 Hz. Los hashes registrados en ella coinciden con los **353 WAV
actuales**, y sus duraciones coinciden con las de esos archivos. El resumen
anterior indica que pasaron de 174 transcripciones vacías a ninguna; esto no
mide exactitud del texto ni sustituye una revisión humana o WER.

Se usaron exclusivamente las nueve variables acordadas:
`word_count`, `words_per_second`, `lexical_diversity`, `filler_count`,
`filler_rate`, `confusion_phrase_count`, `self_correction_count`,
`repetition_count`, `repair_count`.
No entran etiqueta, ID, split, nombre de archivo, hash, estado/error de ASR,
modelo de ASR, versión del pipeline, tiempo de procesamiento ni tasas de
muestreo. La duración real del audio solo se usa para calcular la tasa de
palabras por segundo de llamada completa; no es velocidad de habla activa.

## Auditoría de generalización

Se calculó SHA-256 del archivo completo, del PCM estéreo sin cabecera y del
PCM del canal del cliente (incluyendo frecuencia de muestreo en los hashes
PCM). **Cero grupos duplicados**, tanto entre splits como dentro de ellos,
en las tres comprobaciones. Esto no detecta recortes, cambios de amplitud,
recodificación ni otros duplicados aproximados; tampoco identifica hablantes.

Se inspeccionaron las cabeceras de los CSV del manifest, transcripciones,
características, reportes y revisiones, además de las claves de los JSON de
turnos. No se encontraron identificadores de hablante o voz. `anon_id` y
`file_name` identifican llamadas; `channel` identifica el lado de la
conversación. No se usaron nombres del texto como identidades o grupos.
El README declara separación por hablante entre train y val, pero no aporta
las identidades necesarias para reproducir esa separación **dentro de train**.
Una división aleatoria por llamada no garantiza separación por hablante/voz.

## Protocolo fijo y resultados internos

Una sola ejecución de StratifiedKFold de cinco particiones, shuffle=True,
seed=42, exclusivamente sobre train. Se compararon únicamente dos modelos:
logística lingüística corregida y logística acústica de referencia. Ambos
usan C=1, max_iter=2000, sin ponderación de clases y umbral 0.5, como la
comparación ya acordada. No hubo búsqueda de parámetros, selección repetida,
nuevos ajustes de variables ni evaluación nueva sobre val.

En cada partición se creó un Pipeline nuevo: imputación por mediana con
indicadores, escalado y regresión logística, ajustados **solo con las filas
de entrenamiento de esa partición**. También se reentrenó la acústica en
cada partición; no se evaluó el artefacto entrenado con todo train sobre sus
propias filas. Para la acústica se usaron las 31 características ya guardadas.
Se guardaron IDs de ajuste/evaluación y estadísticas de imputación/escalado
por partición para auditarlo.

Resultados agregados de las predicciones fuera de partición de 282 llamadas:

| Modelo | Accuracy | F1 synthetic | ROC AUC | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lingüístico corregido | 0.787234 | 0.827586 | 0.867152 | 0.457287 | 0.146966 |
| Acústico reentrenado por partición | 0.985816 | 0.988166 | 0.998639 | 0.050199 | 0.011585 |

Media ± desviación entre particiones: accuracy lingüística
0.787343 ± 0.030797; acústica 0.985840 ± 0.007082. AUC lingüística
0.865899 ± 0.036463; acústica 0.999733 ± 0.000535. La AUC media por partición
y la AUC agregada son estadísticas distintas. La dispersión entre cinco
particiones no constituye un intervalo de confianza de generalización.

Matrices de confusión (filas reales, columnas predichas; human/synthetic):
lingüística `[[78, 35], [25, 144]]`; acústica `[[111, 2], [2, 167]]`.

No se compararon fusiones ni se entrenó un segundo nivel. El CSV de
predicciones internas es un registro de evaluación de modelos base; no debe
reutilizarse como evaluación independiente de una fusión. Una futura
comparación de fusión requerirá particiones externas y predicciones internas
generadas reentrenando todos los modelos base exclusivamente dentro de cada
partición externa de entrenamiento.

Estos resultados no demuestran separación por hablante ni ausencia de
sobreajuste. Las variables y configuraciones tienen una historia previa de
exploración. Las métricas anteriores de val (incluida la comparación
acústica/lenguaje/combinación de `LENGUAJE.md`) son **exploratorias**, porque
val ya fue observado. No se recalcularon ni se usaron para decidir esta revisión.

## Artefactos y reproducción

Nuevo modelo, ajustado finalmente con las 282 llamadas de train:

`models/language_corrected_logistic_20260913T101639050249Z.joblib`

Incluye Pipeline, orden explícito de variables, etiquetas, umbral, IDs de
train y hash de la fuente. Se verificó que recargarlo reproduce sus
probabilidades. Está ignorado por Git, igual que los otros modelos: existe
localmente y no se distribuirá mediante un commit de código automáticamente.

Resultados: `review/corrected_model_20260913T101639050249Z/`:

- `report.json`: configuración, métricas por partición/agregadas, hashes de modelos y limitaciones.
- `audio_hashes.csv`, `duplicates.json`: auditoría de los 353 audios.
- `metadata_inventory.json`: esquemas revisados y ausencia de grupos de hablante.
- `language_features_verified.csv`: copia verificada de las características.
- `train_oof_predictions.csv`: predicciones internas exclusivamente de train.
- `language_fold_audit.json`, `acoustic_fold_audit.json`: trazabilidad de particiones y preprocesamiento.
- `provenance.json`: hashes de los archivos previos preservados.

```bash
.venv/bin/python -m src.language.review_corrected_model
.venv/bin/python -m unittest discover -s tests -p 'test_corrected_model_review.py' -v
```

Cada ejecución de revisión crea nombres nuevos con timestamp y se niega a
sobrescribir el nuevo artefacto. No es necesario repetir la evaluación para
usar los resultados guardados. Tres pruebas pasaron: exclusión de las filas
retenidas en imputación/escalado, rechazo de val en la validación interna y
rechazo de inconsistencias de etiquetas/cobertura.

## Decisión operativa y pendientes

Se conserva el acústico: la logística lingüística aislada tiene peor resultado
interno, y esta revisión no aporta evidencia de beneficio incremental de una
fusión ni de generalización a voces nuevas. No se cambia el modelo servido.

Para usar lenguaje desde un WAV recibido falta conectar y probar ASR interno
con el mismo remuestreo/canal/configuración de entrenamiento, carga única,
tratamiento de errores y presupuesto medido de latencia/memoria. Existe el
transcriptor batch corregido, pero `/detect` no lo ejecuta. No se acepta usar
transcripciones externas o buscar una llamada conocida por su ID.

El predictor conductual actual requiere un JSON de turnos. Falta extraerlos
del audio recibido y verificar que esa extracción reproduce las variables
usadas al entrenar. Por ahora no cumple el requisito operativo para servirlo.
Una fusión con esas modalidades hereda ambos pendientes y necesita evaluación
anidada completa, sin elegir ajustes a partir de val.

También faltan grupos de hablante/voz verificables para validación agrupada,
evaluación independiente y revisión humana de calidad de ASR. No se intentó
identificar a las personas del dataset.

## Git

Trabajo sobre `feature/behavior-leo`, HEAD `a3bb91a`. Se conservaron todos los
cambios locales anteriores. Esta revisión añadió el script, sus tests, este
documento, el directorio de resultados y un modelo local ignorado. No se hizo
commit, push, merge ni eliminación en esta revisión. El stash de respaldo de
la integración anterior se conserva.
