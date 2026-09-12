import os
import pandas as pd

# Matriz de intención basada en reglas (Rule-Based Classification):
# Mapeamos palabras clave habituales en atención al cliente con su categoría correspondiente.
INTENT_KEYWORDS = {
    "reclamo_queja": [
        "mala atencion", "molesto", "no funciona", "pesimo", "error", 
        "queja", "cobro indebido", "falla", "servicio terrible", "cancelar"
    ],
    "soporte_tecnico": [
        "no enciende", "falla tecnica", "configurar", "resetear", 
        "conexion", "clave", "contraseña", "reparar", "ayuda tecnica"
    ],
    "consulta_informacion": [
        "precio", "costo", "horario", "informacion", "donde esta", 
        "requisitos", "planes", "saldo", "cuanto cuesta"
    ],
    "seguimiento_tramite": [
        "status", "estatus", "folio", "mi pedido", "seguimiento", 
        "cuando llega", "estado de mi cuenta"
    ]
}

# Léxicos de sentimiento (Palabras de polaridad positiva y negativa)
POSITIVE_WORDS = ["excelente", "bueno", "gracias", "perfecto", "amable", "solucionado", "rapido"]
NEGATIVE_WORDS = ["mal", "pesimo", "incompetente", "terrible", "molesto", "injusto", "fraude", "tardado"]



# FUNCIONES DE LÓGICA DE NEGOCIO

def classify_intent(text):
    """
    Determina la intención principal del usuario mediante conteo de palabras clave.
    
    Proceso:
    1. Convierte la transcripción a minúsculas para ignorar mayúsculas/minúsculas.
    2. Cuenta cuántas palabras de cada categoría de intención aparecen en el texto.
    3. Retorna la categoría con mayor número de coincidencias.
    4. Si no encuentra ninguna coincidencia, asigna 'otro_general'.
    """
    text_lower = str(text).lower()
    scores = {intent: 0 for intent in INTENT_KEYWORDS}
    
    # Recorrer cada categoría y sus palabras clave buscando coincidencias en el texto
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[intent] += 1
                
    max_score = max(scores.values())
    
    # Si ninguna palabra clave coincidió, cae en la categoría genérica
    if max_score == 0:
        return "otro_general"
    
    # Retorna la clave (categoría) que tuvo el mayor puntaje
    return max(scores, key=scores.get)


def analyze_sentiment(text):
    """
    Evalúa la polaridad emocional del cliente (positivo, negativo, neutro).
    
    Proceso:
    1. Transforma el texto a minúsculas.
    2. Cuenta la presencia de términos positivos vs. términos negativos.
    3. Compara ambas cantidades para determinar la tendencia emocional dominante.
    """
    text_lower = str(text).lower()
    
    # Contar cuántas palabras de cada polaridad están presentes en la llamada
    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    
    # Clasificación por dominancia de puntuación
    if neg_count > pos_count:
        return "negativo"
    elif pos_count > neg_count:
        return "positivo"
    
    # Si pos_count == neg_count (o ambos son 0)
    return "neutro"


def process_analysis():
    """
    Función principal de orquestación del análisis de lenguaje:
    1. Carga el CSV producido por el módulo de transcripción (transcribe.py).
    2. Aplica la clasificación de intención y sentimiento a cada llamada.
    3. Guarda los resultados enriquecidos en un nuevo CSV.
    """
    # Construcción dinámica de rutas para evitar problemas entre macOS / Windows
    # Subimos 3 niveles desde src/language/intent_sentiment.py hasta la raíz del proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv = os.path.join(project_root, "src", "language", "caller_transcriptions.csv")
    output_csv = os.path.join(project_root, "src", "language", "classified_transcriptions.csv")
    
    # Verificar que el CSV de transcripciones exista antes de continuar
    if not os.path.exists(input_csv):
        print(f"Error: No se encuentra {input_csv}. Asegúrate de que transcribe.py terminó de ejecutarse.")
        return
        
    # Leer el CSV con Pandas
    df = pd.read_csv(input_csv)
    
    # Reemplazar valores nulos (NaN) por texto vacío para evitar fallos de lectura
    df['caller_transcript'] = df['caller_transcript'].fillna("")
    
    print("Iniciando análisis NLP de intención y sentimiento...")
    
    # .apply() ejecuta la función fila por fila sobre la columna 'caller_transcript'
    df['intent'] = df['caller_transcript'].apply(classify_intent)
    df['sentiment'] = df['caller_transcript'].apply(analyze_sentiment)
    
    # Guardar la nueva estructura en el CSV de salida
    df.to_csv(output_csv, index=False)
    print(f"¡Análisis NLP finalizado con éxito! Resultados guardados en:\n{output_csv}")


# Punto de entrada estándar para cuando el script se ejecuta directamente en terminal
if __name__ == "__main__":
    process_analysis()