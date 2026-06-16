# En una terminal o notebook
import pandas as pd
df = pd.read_csv('data/processed/features_por_artista.csv')
print(f"Shape: {df.shape}")
print(f"\nPrimeras 3 filas:\n{df.head(3)}")
print(f"\nRango de valores (muestra):\n{df[['danceability', 'energy', 'loudness', 'decade']].describe()}")