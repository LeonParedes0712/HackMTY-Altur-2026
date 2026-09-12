import os
import re
import pandas as pd

# lexico y patrones regulares de reflejo conversacional
fillers = [
    r'\beh+\b', r'\bem+\b', r'\bmmm+\b', r'\beste\b', 
    r'\bpues\b', r'\bbueno\b', r'\ba ver\b'
]

confusion_phrases = [
    r'\bqué\b', r'\bcómo\b', r'\bperdón\b', r'otra vez', 
    r'no entendí', r'me repites', r'\bcuál\b', r'cómo dice'
]

self_corrections = [
    r'\bperdón\b', r'\bdigo\b', r'más bien', r'\bespera\b'
]

repair_phrases = [
    r'\bcómo\b', r'\bperdón\b', r'no entendí', r'otra vez', 
    r'a ver', r'espérame', r'no, espera', r'me equivoqué'
]

def extract_language_features(text, duration_sec):
    """extrae características de duda, repetición y diversidad léxica."""
    if not isinstance(text, str) or not text.strip():
        return {
            "word_count": 0,
            "words_per_second": 0.0,
            "lexical_diversity": 0.0,
            "filler_count": 0,
            "filler_rate": 0.0,
            "confusion_phrase_count": 0,
            "self_correction_count": 0,
            "repetition_count": 0,
            "repair_count": 0
        }
    
    text_clean = text.lower()
    words = re.findall(r'\b\w+\b', text_clean)
    word_count = len(words)
    unique_words = len(set(words))
    
    # diversidad léxica y velocidad del habla
    lexical_diversity = round(unique_words / word_count, 4) if word_count > 0 else 0.0
    words_per_second = round(word_count / duration_sec, 4) if (duration_sec and duration_sec > 0) else 0.0
    
    # conteos de muletillas (fillers) y tasa por cada 100 palabras
    filler_count = sum(len(re.findall(pat, text_clean)) for pat in fillers)
    filler_rate = round((filler_count / word_count) * 100, 2) if word_count > 0 else 0.0
    
    # conteos de reflejo conversacional
    confusion_count = sum(len(re.findall(pat, text_clean)) for pat in confusion_phrases)
    self_corr_count = sum(len(re.findall(pat, text_clean)) for pat in self_corrections)
    repair_count = sum(len(re.findall(pat, text_clean)) for pat in repair_phrases)
    
    # repetición de palabras consecutivas (ej. "sí sí", "no no", "este este")
    repetition_count = len(re.findall(r'\b(\w+)\s+\1\b', text_clean))

    return {
        "word_count": word_count,
        "words_per_second": words_per_second,
        "lexical_diversity": lexical_diversity,
        "filler_count": filler_count,
        "filler_rate": filler_rate,
        "confusion_phrase_count": confusion_count,
        "self_correction_count": self_corr_count,
        "repetition_count": repetition_count,
        "repair_count": repair_count
    }

if __name__ == "__main__":
    # definición de rutas principales
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    transcripts_path = os.path.join(project_root, "src", "language", "caller_transcriptions.csv")
    metrics_path = os.path.join(project_root, "src", "audio", "audio_metrics.csv")
    
    print("cargando transcripciones y métricas acústicas...")
    df_trans = pd.read_csv(transcripts_path)
    df_audio = pd.read_csv(metrics_path)
    
    # unir la transcripción con la duración del audio para calcular words_per_second
    df = pd.merge(df_trans, df_audio[['file_name', 'duration_sec']], on='file_name', how='left')
    
    rows = []
    for _, row in df.iterrows():
        features = extract_language_features(
            text=row.get('caller_transcript', ''), 
            duration_sec=row.get('duration_sec', 0)
        )
        # usamos anon_id conservando el identificador del archivo
        features['anon_id'] = row['file_name']
        rows.append(features)
        
    out_df = pd.DataFrame(rows)
    
    # ordenar las columnas exactamente como lo requiere el equipo
    ordered_cols = [
        'anon_id',
        'word_count',
        'words_per_second',
        'lexical_diversity',
        'filler_count',
        'filler_rate',
        'confusion_phrase_count',
        'self_correction_count',
        'repetition_count',
        'repair_count'
    ]
    out_df = out_df[ordered_cols]
    
    # exportar resultados a outputs/language_features.csv
    out_dir = os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "language_features.csv")
    
    out_df.to_csv(out_path, index=False)
    print(f"éxito! features de lenguaje procesadas para {len(out_df)} llamadas.")
    print(f"archivo guardado en: {out_path}")