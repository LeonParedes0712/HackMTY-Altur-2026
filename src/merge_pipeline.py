import os
import pandas as pd


# PIPELINE DE CONSOLIDACIÓN Y FUSIÓN DE DATOS (DATA MERGING)


def merge_results():
    """
    Función principal de integración:
    1. Localiza los CSV de transcripciones/NLP (src/language/) y de métricas acústicas (src/audio/).
    2. Realiza un cruce de tablas (Inner Join) utilizando 'file_name' como clave primaria.
    3. Exporta la estructura unificada final a 'final_altur_report.csv' en la raíz del proyecto.
    """
   
    # 1. RESOLUCIÓN DINÁMICA DE RUTAS DEL PROYECTO
    # Subimos 2 niveles desde 'src/merge_pipeline.py' para obtener la carpeta raíz:
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Rutas absolutas hacia los archivos CSV de los submódulos de trabajo
    transcriptions_file = os.path.join(project_root, "src", "language", "classified_transcriptions.csv")
    metrics_file = os.path.join(project_root, "src", "audio", "audio_metrics.csv")
    
    # Ruta de salida final en la raíz del proyecto
    output_final = os.path.join(project_root, "final_altur_report.csv")
    
    
    # 2. VALIDACIÓN DE PRERREQUISITOS
    # Verifica que ambos procesos previos (transcripción + NLP y métricas de audio) hayan generado su CSV
    if not os.path.exists(transcriptions_file):
        print(f"Error: Falta el archivo {transcriptions_file}")
        print("-> Asegúrate de ejecutar 'transcribe.py' e 'intent_sentiment.py' primero.")
        return

    if not os.path.exists(metrics_file):
        print(f"Error: Falta el archivo {metrics_file}")
        print("-> Asegúrate de ejecutar 'metrics.py' primero.")
        return
        
    # 3. LECTURA Y FUSIÓN DE DATOS CON PANDAS
    
    print("Cargando archivos de resultados parciales...")
    df_trans = pd.read_csv(transcriptions_file)
    df_metrics = pd.read_csv(metrics_file)
    
    print("Cruzando información por 'file_name' (Inner Join)...")
    # Une df_trans (file_name, caller_transcript, intent, sentiment) 
    # con df_metrics (file_name, duration_sec, rms_energy, silence_percentage)
    final_df = pd.merge(df_trans, df_metrics, on="file_name", how="inner")
    
    
    # 4. EXPORTACIÓN DEL REPORTE FINAL CONSOLIDADO
    
    # Guardamos el DataFrame resultante sin la columna de índice predeterminada de pandas (index=False)
    final_df.to_csv(output_final, index=False)
    
    print("\n" + "="*70)
    print("¡PIPELINE EJECUTADO CON ÉXITO!")
    print(f"Filas consolidadas: {len(final_df)}")
    print(f"Reporte final guardado en:\n   {output_final}")
    print("="*70 + "\n")

# Punto de entrada estándar de ejecución
if __name__ == "__main__":
    merge_results()