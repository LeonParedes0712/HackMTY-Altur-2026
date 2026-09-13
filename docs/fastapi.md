# Servidor final

Desde la raíz del repositorio:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Swagger: http://localhost:8000/docs. `POST /detect-upload` recibe el campo
multipart `file`. `GET /health` confirma la carga del modelo.
`POST /detect` recibe `{"audio":"<base64>"}` y devuelve
`{"is_synthetic":true}` (o `false`). Ambos usan el mismo predictor acústico,
con umbral fijo 0.5 y canal 0; no convierten ni mezclan los canales.

El README del reto, sección Evaluation, declara `confidence` opcional y usado
para desempate/calibración, pero no define si es probabilidad de synthetic o
de la clase elegida. Por la instrucción final del usuario se omite. Si los
organizadores confirman la segunda interpretación, correspondería usar
`p_synthetic` cuando la predicción sea synthetic y `p_human` en otro caso.

Se exige base64 estricto, WAV PCM de 16 bits, 8000 Hz, dos canales y datos
completos. El límite del archivo es 30 MiB; el del cuerpo HTTP es 32 MiB (incluye base64 o multipart). La duración
admitida es mayor que cero y hasta 600 s, sujeta también al tamaño total.
Son los límites operativos recuperados de la rama de transcripciones;
no son límites publicados por el reto. La validación WAV se comparte en
`src/acoustic/validation.py`. Se rechaza el canal del cliente completamente silencioso. Errores: 400 (base64/duración/truncamiento), 415 (formato),
413 (tamaño) y 422 (estructura de entrada).

El modelo `models/acoustic_logistic.pkl` se carga una vez en el lifespan por
proceso; usar un worker para una sola instancia. Los audios temporales se
limpian al terminar cada predicción. Los errores de estructura no reflejan
el cuerpo del usuario. No se envía audio a servicios externos.

## Verificación

```bash
.venv/bin/python -m pip install -r requirements-test.txt
NUMBA_CACHE_DIR=/tmp/hackmty26-numba .venv/bin/python -m unittest discover -s tests -p test_api.py -v
NUMBA_CACHE_DIR=/tmp/hackmty26-numba .venv/bin/python tests/smoke_uvicorn.py
```

Las pruebas usan las primeras llamadas human y synthetic de `train`, sin
seleccionarlas según aciertos. Comprueban igualdad entre JSON y upload,
contrato, health, Swagger, carga única y validaciones. La segunda prueba
arranca y detiene Uvicorn con HTTP real sobre localhost:8766.

Métricas calculadas con el artefacto existente y las características acústicas
ya guardadas de las 71 llamadas de `val`, sin reentrenar ni ajustar el umbral:

| Métrica | Valor |
| --- | --- |
| Accuracy | 0.971831 |
| F1 synthetic | 0.971429 |
| Log loss | 0.107372 |
| ROC AUC | 1.000000 |

Matriz de confusión (filas reales, columnas predichas; human, synthetic):
`[[35, 2], [0, 34]]`. Estas métricas proceden de características almacenadas,
no de reprocesar todo `val` por HTTP. No garantizan generalización ni ausencia
de sobreajuste. El código de entrenamiento ajusta imputación, escalado y
clasificador exclusivamente con `train`; no se ha auditado toda la historia
de selección del artefacto. No se entrenó ni evaluó una fusión en este cambio.

## Pendientes y servidor anterior

- Confirmar con los organizadores la semántica de confidence y los límites
  de duración/tamaño, si requieren otros valores.
- Medir latencia/concurrencia en el equipo de despliegue; extracción de pitch
  consume CPU y la primera inferencia puede compilar código de Numba.
- Verificar generalización con evaluación independiente; si se evalúa fusión
  mediante CV, cada partición debe entrenar también sus modelos base, sin
  reutilizar una tabla OOF global como evaluación independiente.
- `src/acoustic/serve.py` se recuperó al integrar la rama de transcripciones.
  Se conserva como referencia histórica y para sus pruebas; su entrada
  ejecutable indica usar Uvicorn. FastAPI es el único servidor final.

Se preservaron los cambios locales de features/predict y los archivos nuevos
de entrenamiento de lenguaje/fusión. Se integró `origin/feature/transcripciones-corregidas-y-detectores` en
`feature/behavior-leo`, después de comparar las ramas y respaldar los cambios
locales. No se hizo push ni merge a main.

Resultado después de integrar las ramas: las 24 pruebas de unittest pasaron
(5.504 s, con caché disponible) y la prueba HTTP real de Uvicorn terminó correctamente.
La llamada humana devolvió `false` y la sintética `true` en ambos endpoints.
`pip check`, compilación de Python y `git diff --check` pasaron.
TestClient emitió un aviso de deprecación de su adaptador httpx; sigue
funcionando en las versiones fijadas. El sandbox restringe sockets: las
pruebas se completaron con la autorización automática para ejecución local.

## Entrega congelada

El artefacto acústico de 4,329 bytes se preparó para incluirse en Git mediante
una excepción específica en `.gitignore`. El modelo y el umbral siguen intactos.
La instalación limpia con Python 3.14.4 y `requirements-test.txt` pasó, al igual
que HTTP real sobre una copia aislada del código y el artefacto. Whisper solo
figura en `requirements-research.txt`, opcional. Véase `docs/delivery.md` para
la lista de commit y comprobaciones finales. No se reentrenó ni transcribió.
