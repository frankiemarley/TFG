import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from music_tfg.paths import RAW_DATA_DIR, EDA_OUTPUTS_DIR, ensure_directories 

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

def load_tracks(path: Path) -> pd.DataFrame:
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
    
def describe_audio_features(df: pd.DataFrame):
    print("Resumen estadístico de las características de audio:")
    return df[AUDIO_FEATURES].describe()

def plot_audio_features(df: pd.DataFrame) -> None:
    """Guarda un grid 3x4 con histogramas de todas las features de audio."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, feature in enumerate(AUDIO_FEATURES):
        axes[i].hist(df[feature].dropna(), bins=50, color="steelblue", edgecolor="black")
        axes[i].set_title(feature)
        axes[i].set_ylabel("Frecuencia")

    # 11 features en grid 3x4 → sobra 1 hueco
    axes[-1].axis("off")

    plt.suptitle("Distribución de features de audio (tracks.csv)", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(EDA_OUTPUTS_DIR / "audio_features_hist.png", dpi=100)
    plt.close()

def extract_year(df: pd.DataFrame) -> pd.DataFrame:
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    return df

def plot_year_distribution(df: pd.DataFrame) -> None:
    """Guarda histograma de años de release."""
    plt.figure(figsize=(10, 5))
    df["release_year"].dropna().hist(bins=50, color="steelblue", edgecolor="black")
    plt.title("Distribución de años de lanzamiento")
    plt.xlabel("Año")
    plt.ylabel("Número de canciones")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUTS_DIR / "release_year_distribution.png", dpi=100)
    plt.close()
    
def artist_stats(df: pd.DataFrame) -> None:
    """Estadísticas sobre artistas en tracks."""
    total_tracks = len(df)
    multi_artist_tracks = (df["n_artists"] > 1).sum()
    unique_artists = df["artists"].explode().nunique()

    print(f"Total de tracks: {total_tracks:,}")
    print(f"Tracks con múltiples artistas: {multi_artist_tracks:,} "
          f"({100 * multi_artist_tracks / total_tracks:.1f}%)")
    print(f"Artistas únicos: {unique_artists:,}")

    print("\nDistribución del número de artistas por track:")
    print(df["n_artists"].value_counts().sort_index().head(10))

    print("\nTop 10 artistas con más canciones:")
    print(df["artists"].explode().value_counts().head(10))

def parse_artists(df: pd.DataFrame) -> pd.DataFrame:
    df['artists'] = df['artists'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df['n_artists'] = df['artists'].apply(len)
    return df

def main()-> None:
    ensure_directories()
    
    tracks_df = load_tracks(RAW_DATA_DIR / "tracks.csv")
    
    print("=" * 60)
    print("EDA TRACKS")
    print("=" * 60)

    basic_info(tracks_df)

    print("\n--- Nulos ---")
    null_summary(tracks_df)

    tracks_df = extract_year(tracks_df)
    print(f"\nNulos en 'release_year' tras parseo: {tracks_df['release_year'].isnull().sum()}")

    print("\n--- Audio features ---")
    print(describe_audio_features(tracks_df))

    plot_audio_features(tracks_df)
    plot_year_distribution(tracks_df)

    tracks_df = parse_artists(tracks_df)
    print("\n--- Artistas ---")
    artist_stats(tracks_df)

    print(f"\nGráficas guardadas en {EDA_OUTPUTS_DIR}")
    
    
if __name__ == "__main__":
    main()