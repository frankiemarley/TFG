import ast
from pathlib import Path

from matplotlib.pylab import empty
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm

from music_tfg.paths import RAW_DATA_DIR, EDA_OUTPUTS_DIR, ensure_directories 

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

def load_artists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def basic_info(df: pd.DataFrame):
    print("Número de filas y columnas:", df.shape)
    print("\nTipos de datos:")
    print(df.dtypes)        

    print(df.columns.tolist())
    
def null_summary(df: pd.DataFrame):
    null_counts = df.isnull().sum()
    print("Número de valores nulos por columna:")
    print(null_counts)
    print(null_counts[null_counts > 0])
    
def parse_genres(df: pd.DataFrame) -> pd.DataFrame:
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    return df

def genre_stats(df: pd.DataFrame):
    all_genres = df['genres'].explode()
    genre_counts = all_genres.value_counts()
    print("Géneros más comunes:")
    print(genre_counts.head(30))
    df["n_genres"] = df["genres"].apply(len)
    empty = (df["n_genres"] == 0).sum()
    print(f"Artistas sin género: {empty:,} ({100 * empty / len(df):.1f}%)")
    return genre_counts

def plot_top_genres(genre_counts: pd.Series, top_n: int = 20):
    top_genres = genre_counts.head(top_n)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_genres.values, y=top_genres.index)
    plt.title(f"Top {top_n} Géneros de Artistas")
    plt.xlabel("Número de Artistas")
    plt.ylabel("Género")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUTS_DIR / "top_genres.png", dpi=100)
    plt.close()
    
def popularity_stats(df: pd.DataFrame):
    print("Resumen estadístico de los seguidores:")
    followers_stats = df['followers'].describe()
    print(followers_stats)
    print("Resumen estadístico de la popularidad:")
    print(df['popularity'].describe())

def plot_popularity_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['popularity'].dropna(), bins=20, kde=True)
    plt.title("Distribución de la popularidad")
    plt.xlabel("Popularidad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUTS_DIR / "popularity_distribution.png")
    plt.close()
    
def main()-> None:
    print("Iniciando análisis de artistas...")
    ensure_directories()
    print(f"Cargando datos desde {RAW_DATA_DIR / 'artists.csv'}")
    artists_df = load_artists(RAW_DATA_DIR / "artists.csv")
    
    print("\n" + "=" * 50)
    print("INFORMACIÓN BÁSICA")
    print("=" * 50)
    basic_info(artists_df)
    
    print("\n" + "=" * 50)
    print("VALORES NULOS")
    print("=" * 50)
    null_summary(artists_df)
    
    print("\nProcesando géneros...")
    artists_df = parse_genres(artists_df)
    
    print("\n" + "=" * 50)
    print("ESTADÍSTICAS DE GÉNEROS")
    print("=" * 50)
    genre_counts = genre_stats(artists_df)
    
    print("\nGenerando gráfico de géneros...")
    plot_top_genres(genre_counts)
    
    print("\n" + "=" * 50)
    print("ESTADÍSTICAS DE POPULARIDAD")
    print("=" * 50)
    popularity_stats(artists_df)
    
    print("\nGenerando gráfico de popularidad...")
    plot_popularity_distribution(artists_df)
    
    print("\nEDA de artistas completado. Gráficos guardados en:", EDA_OUTPUTS_DIR)
   
if __name__ == "__main__":
    main()
