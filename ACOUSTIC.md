# Parte acústica — Mayorga

Extrae 36 características del canal 0, una fila por llamada, sin mezclar al agente.
La primera columna es `anon_id`; incluye `label` y `split` del manifest.
Se procesaron 353 llamadas reales sin errores: 282 train y 71 val. Los audios originales permanecen fuera de Git.

## Ejecución desde esta carpeta

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-acoustic.txt
python -m src.acoustic.inspect_audio --data /ruta/al/dataset
python -m src.acoustic.features --data /ruta/al/dataset
python -m src.acoustic.train
python -m unittest discover -s tests
```

El dataset contiene `manifest.csv`, `audio/<anon_id>.wav` y, para el modo turns,
`turns/<anon_id>.json`. No subir audios ni datos derivados sin revisar los términos de Altur.

## Segmentación

Por defecto `--mode energy` usa un detector sencillo de actividad por energía,
que también puede ejecutarse en llamadas nuevas sin archivos JSON. Puede confundir
ruido con voz o perder habla débil; revisar con audios reales antes de evaluarlo.
Para experimentar con los turnos proporcionados:

```sh
python -m src.acoustic.features --data /ruta/al/dataset --mode turns --output outputs/turns_features.csv
python -m src.acoustic.train --features outputs/turns_features.csv --output outputs/turns_model
```

Un modelo entrenado con turns requiere turns al predecir. Para integrar con `/detect`,
usar el modelo energy o proporcionar exactamente el mismo segmentador al entrenar
y predecir. El modo se guarda junto al modelo. No se afirma que ambos modos sean equivalentes.

## Definiciones

- WAV estéreo PCM de 16 bits, 8000 Hz. Solo canal 0; muestras / 32768.
- Sin normalizar por pico, quitar ruido ni cambiar pitch.
- Se recortan y fusionan intervalos superpuestos. Cada segmento se enmarca por separado:
  ventanas de 25 ms, paso de 10 ms; se omiten fragmentos menores a 25 ms.
- MFCC: FFT 256, Hann, 26 filtros mel entre 0–4000 Hz, log natural, DCT-II ortonormal.
  `mfcc_1` corresponde a C0, hasta `mfcc_13` = C12. Media y desviación poblacional.
- Energía: media del cuadrado de las muestras por ventana, no RMS ni dB.
- ZCR: proporción de cambios de signo; centroid y rolloff (85% de potencia) en Hz.
- Agregación por ventanas: segmentos largos aportan más observaciones.
- Pitch queda fuera de esta primera versión.

`caller_duration_s`, `status`, `segmentation`, `anon_id`, `label` y `split` NO entran al modelo.
Silencio o ausencia de ventanas útiles produce NaN y `no_usable_speech`; errores de archivos
se registran en `.errors.json`. Se conserva una fila por llamada. El entrenamiento exige
resolver esos casos y no los elimina silenciosamente.

## Modelo y salida

Logistic Regression con escalado ajustado únicamente en train. Alternativa:
`python -m src.acoustic.train --model forest --output outputs/forest`.
Validación solo en val: accuracy, precision, recall, F1, ROC-AUC, Brier y matriz de confusión
con orden human/synthetic. Umbral fijo 0.5. ROC-AUC no se presenta como métrica oficial.

`outputs/acoustic_scores.csv` contiene exactamente `anon_id,acoustic_score`;
el score es P(synthetic). Incluye ambas particiones: unir por `anon_id` con manifest.
**Los scores de train son in-sample: NO utilizarlos para entrenar la fusión del equipo.**
Para esa etapa generar predicciones out-of-fold dentro de train, idealmente agrupadas por
hablante si el equipo dispone de esos identificadores. No usar val para entrenar la fusión.
No se afirma que las probabilidades estén calibradas.

Integración en Python:

```python
from src.acoustic.train import predict_call
score = predict_call('nueva.wav', 'outputs/acoustic_model.joblib')
```

Esta función carga el modelo por llamada; el servicio definitivo debería cargarlo una vez
al arrancar. La integración decide cómo tratar falta de voz y cómo traducir el score al
campo `confidence` según el contrato que confirme Altur. No se implementa aquí `/detect`.

## Resultado inicial (2026-09-12)

Modelo logístico, segmentación energy y umbral 0.5, sin ajustar con val:
accuracy 0.957746, precision 0.918919, recall 1.0, F1 0.957746,
ROC-AUC 0.996820 y Brier 0.032090. Matriz [[34, 3], [0, 34]]:
3 humanos marcados como sintéticos; ninguna sintética omitida en esta validación.
No son resultados sobre el test oculto ni una garantía de generalización.

Se generó también `outputs/acoustic_features_turns.csv`, sin entrenar un segundo modelo.
`outputs/acoustic_train_summary.json` compara medias por clase usando solo train.
La verificación de formato de 5 llamadas por clase pasó; la escucha manual por el equipo
sigue pendiente. Las cinco pruebas automáticas pasaron y una predicción con el modelo
guardado coincidió con su CSV. Dependencias exactas en `requirements-acoustic-lock.txt`.
