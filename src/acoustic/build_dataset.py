from pathlib import Path
import pandas as pd

from features import extract_features


ROOT = Path(__file__).resolve().parents[2]

# Leer el manifest
df = pd.read_csv(ROOT / "manifest.csv")

# Tomar 5 audios de cada clase para probar
test_df = pd.concat([
    df[df["label"] == "human"].head(5),
    df[df["label"] == "synthetic"].head(5)
])

rows = []

for _, row in test_df.iterrows():
    anon_id = row["anon_id"]
    label = row["label"]

    audio_path = ROOT / "audio" / f"{anon_id}.wav"

    # Verificar que el audio exista
    if not audio_path.exists():
        print("No existe:", audio_path)
        continue

    # Sacar características usando la función de tu compañero
    features = extract_features(audio_path)

    # Agregar información del manifest
    features["anon_id"] = anon_id
    features["label"] = label

    rows.append(features)

# Convertir todos los resultados a DataFrame
result_df = pd.DataFrame(rows)

# Crear carpeta outputs si no existe
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

# Ruta del CSV final
output_path = output_dir / "acoustic_features.csv"

# Guardar resultados
result_df.to_csv(output_path, index=False)

print("Guardado en:", output_path)
print(result_df)