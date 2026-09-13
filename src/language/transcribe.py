"""Transcripción del caller a 16 kHz, con checkpoints y comparación."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
TARGET_SR = 16000
VERSION = "caller-resample-v1"
FIELDS = ["file_name", "caller_transcript", "status", "error", "word_count",
          "source_sample_rate", "target_sample_rate", "duration_sec", "elapsed_sec",
          "audio_sha256", "model", "pipeline_version"]


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def prepare_audio(file_path):
    sample_rate, audio = wavfile.read(file_path)
    if sample_rate <= 0 or audio.size == 0:
        raise ValueError("Audio vacío o frecuencia inválida")
    if audio.ndim == 2:
        if audio.shape[1] == 0:
            raise ValueError("Audio sin canales")
        audio = audio[:, 0]
    elif audio.ndim != 1:
        raise ValueError("Se requiere audio mono o multicanal")
    # scipy devuelve PCM 24-bit alineado a la izquierda en int32.
    if np.issubdtype(audio.dtype, np.signedinteger):
        audio = audio.astype(np.float32) / float(2 ** (audio.dtype.itemsize * 8 - 1))
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)
    else:
        raise ValueError(f"Formato de muestras no soportado: {audio.dtype}")
    if not np.isfinite(audio).all():
        raise ValueError("Audio con valores no finitos")
    duration = len(audio) / sample_rate
    if sample_rate != TARGET_SR:
        divisor = math.gcd(sample_rate, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // divisor, sample_rate // divisor)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate, duration


def process_single_audio(file_path, model, model_name):
    start = time.perf_counter()
    result = dict.fromkeys(FIELDS, "")
    result.update(file_name=Path(file_path).name, word_count=0,
                  target_sample_rate=TARGET_SR, model=model_name, pipeline_version=VERSION)
    try:
        result["audio_sha256"] = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
        audio, sr, duration = prepare_audio(file_path)
        result.update(source_sample_rate=sr, duration_sec=round(duration, 4))
        segments, _ = model.transcribe(audio, language="es", beam_size=1)
        # El generador puede fallar al iterarse; también se captura ese error.
        text = " ".join(segment.text.strip() for segment in segments).strip()
        result.update(caller_transcript=text, status="success" if text else "empty",
                      word_count=word_count(text))
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    result["elapsed_sec"] = round(time.perf_counter() - start, 3)
    return result


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as handle:
            temp = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def compare(previous, rows, output):
    old = {row["file_name"]: row for row in previous}
    comparisons = []
    for row in rows:
        prior = old.get(row["file_name"])
        comparisons.append({"file_name": row["file_name"], "previous_available": prior is not None,
                            "old_word_count": word_count(prior["caller_transcript"]) if prior else "",
                            "new_word_count": row["word_count"], "new_status": row["status"],
                            "error": row["error"]})
    matched = [r for r in comparisons if r["previous_available"]]
    summary = {
        "processed": len(rows), "matched_previous": len(matched),
        "old_empty": sum(r["old_word_count"] == 0 for r in matched),
        "new_empty_text_matched": sum(int(r["new_word_count"]) == 0 for r in matched),
        "old_errors": None,
        "old_errors_note": "El CSV anterior no distinguía errores de audio sin texto.",
        "new_success": sum(r["status"] == "success" for r in rows),
        "new_empty": sum(r["status"] == "empty" for r in rows),
        "new_errors": sum(r["status"] == "error" for r in rows),
        "old_words_matched": sum(r["old_word_count"] for r in matched),
        "new_words_matched": sum(int(r["new_word_count"]) for r in matched),
        "models": sorted({r["model"] for r in rows}),
        "note": "Más palabras o menos vacíos no demuestran precisión sin referencia humana.",
    }
    atomic_csv(output.with_suffix(".comparison.csv"), comparisons,
               ["file_name", "previous_available", "old_word_count", "new_word_count", "new_status", "error"])
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    return summary


def run_batch(files, output, previous, model_name="tiny", retry_errors=False, model_factory=None):
    output = Path(output)
    rows = read_csv(output) if output.exists() else []
    by_name = {}
    for row in rows:
        if set(row) != set(FIELDS) or row["status"] not in {"success", "empty", "error"}:
            raise ValueError("El archivo existente no es un checkpoint compatible")
        if row["pipeline_version"] != VERSION or row["model"] != model_name:
            raise ValueError("El checkpoint usa otro modelo o versión; elige otro --output")
        if row["file_name"] in by_name:
            raise ValueError("El checkpoint contiene nombres duplicados")
        by_name[row["file_name"]] = row
    pending = []
    for path in files:
        saved = by_name.get(path.name)
        if saved is not None and not (retry_errors and saved["status"] == "error"):
            if saved["audio_sha256"] and path.exists():
                if saved["audio_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f"El audio cambió: {path.name}; usa otro --output")
            continue
        pending.append(path)
    print(f"Seleccionados: {len(files)} | Pendientes: {len(pending)}", flush=True)
    if pending:
        if model_factory is None:
            from faster_whisper import WhisperModel
            model_factory = WhisperModel
        # Un único modelo para todo el lote, incluido el recorrido de los generadores.
        model = model_factory(model_name, device="cpu", compute_type="int8")
        for index, path in enumerate(pending, 1):
            result = process_single_audio(path, model, model_name)
            by_name[path.name] = result
            atomic_csv(output, list(by_name.values()), FIELDS)
            print(f"[{index}/{len(pending)}] {path.name}: {result['status']} "
                  f"({result['word_count']} palabras) {result['error']}", flush=True)
    summary = compare(previous, list(by_name.values()), output)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="CSV corregido; nunca el CSV anterior")
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--model", default="tiny", help="Modelo Whisper o ruta local")
    parser.add_argument("--limit", type=int, help="Primeras N llamadas seleccionadas, incluidas las ya procesadas")
    parser.add_argument("--sample-balanced", action="store_true", help="Alterna llamadas antes vacías y con texto, más cortas primero")
    parser.add_argument("--retry-errors", action="store_true", help="Reintenta únicamente errores guardados; success/empty se conservan")
    args = parser.parse_args()
    root = args.root.resolve()
    previous_path = (args.previous or root / "src/language/caller_transcriptions.csv").resolve()
    output = (args.output or root / "src/language/caller_transcriptions_corrected.csv").resolve()
    if output == previous_path or output == (root / "src/language/caller_transcriptions.csv").resolve():
        parser.error("--output no puede sobrescribir las transcripciones originales")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser positivo")
    previous = read_csv(previous_path)
    manifest = read_csv(root / "manifest.csv")
    names = [r["anon_id"] for r in manifest]
    if len(names) != len(set(names)):
        parser.error("Manifest contiene IDs duplicados")
    if args.sample_balanced:
        old = {r["file_name"]: r["caller_transcript"].strip() for r in previous}
        ordered = sorted(manifest, key=lambda r: (float(r["duration_s"]), r["anon_id"]))
        empty = [r for r in ordered if old.get(r["anon_id"]+".wav") == ""]
        nonempty = [r for r in ordered if old.get(r["anon_id"]+".wav")]
        if not empty or not nonempty or (args.limit is not None and args.limit < 2):
            parser.error("La muestra requiere al menos una llamada antes vacía y una con texto")
        manifest = []
        for i in range(max(len(empty), len(nonempty))):
            for group in (empty, nonempty):
                if i < len(group):
                    manifest.append(group[i])
    files = [root / "audio" / f"{r['anon_id']}.wav" for r in manifest[:args.limit]]
    # Evita dos procesos escribiendo el mismo checkpoint simultáneamente.
    output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with output.with_suffix(output.suffix+".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Otro proceso está usando este archivo de salida")
        run_batch(files, output, previous, args.model, args.retry_errors)


if __name__ == "__main__":
    main()
