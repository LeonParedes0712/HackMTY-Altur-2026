import os
import glob
import pandas as pd
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
from concurrent.futures import ProcessPoolExecutor

# Función que ejecuta cada núcleo de la CPU de forma independiente
def process_single_audio(file_path):
    try:
        # Cargar el modelo 'tiny' con cuantización int8 (pesa ~75MB, ultrarrápido y consume mínima RAM)
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        
        # 1. Leer la frecuencia de muestreo y el arreglo numérico del audio .wav
        sample_rate, audio_data = wavfile.read(file_path)
        
        # 2. Aislar únicamente el Canal 0 (Cliente/Caller). 
        # Si audio_data tiene 2 dimensiones (estéreo), tomamos la columna 0.
        caller_audio = audio_data[:, 0] if audio_data.ndim > 1 else audio_data
        
        # 3. Convertir los datos PCM de entero de 16 bits a punto flotante (float32) entre -1.0 y 1.0, 
        # que es el formato requerido por la red neuronal de Whisper.
        caller_audio = caller_audio.astype(np.float32) / 32768.0

        # 4. Decodificar la voz a texto en español. 
        # Usar beam_size=1 (búsqueda codiciosa) acelera la inferencia al máximo sin explorar múltiples hipótesis.
        segments, _ = model.transcribe(caller_audio, language="es", beam_size=1)
        
        # 5. Concatenar los fragmentos temporales detectados en una sola cadena de texto limpia
        text = " ".join([segment.text.strip() for segment in segments])
        
        return {
            "file_name": os.path.basename(file_path),
            "caller_transcript": text
        }
    except Exception as e:
        # Si un archivo está corrupto, captura la excepción y evita interrupciones en el lote
        return {
            "file_name": os.path.basename(file_path),
            "caller_transcript": ""
        }

if __name__ == "__main__":
    # Obtener la ruta raíz del proyecto subiendo 3 niveles desde la ubicación actual de este script

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    audio_pattern = os.path.join(project_root, "audio", "*.wav")
    audio_files = glob.glob(audio_pattern)

    total = len(audio_files)
    print(f"Iniciando transcripción ultrarrápida en paralelo para {total} llamadas...\n")

    results = []
    
    # Calcular núcleos de CPU a utilizar (deja 1 núcleo libre para mantener fluida la Mac)
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    
    # ProcessPoolExecutor distribuye la lista de audios entre los múltiples núcleos de la CPU al mismo tiempo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for index, result in enumerate(executor.map(process_single_audio, audio_files), start=1):
            results.append(result)
            print(f"[{index}/{total}] Procesado: {result['file_name']}")

    # Crear una tabla con Pandas y exportarla a un archivo CSV en src language
    output_csv = os.path.join(project_root, "src", "language", "caller_transcriptions.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n¡Completado con éxito! Transcripciones guardadas en: {output_csv}")