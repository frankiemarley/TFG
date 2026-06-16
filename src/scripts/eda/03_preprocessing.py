"""
Bloque 3: Preprocesado.
Construye tabla de features por artista (filtrada y normalizada).
Output: data/processed/features_por_artista.csv
"""

import ast
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

from music_tfg.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, ensure_directories

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

# Parametros de filtrado
MIN_FOLLOWERS = 250 
MIN_SONGS_PER_ARTIST = 5
MIN_POPULARITY = 10
TOP_N_GENRES = 30

def _parse_list_column(series: pd.Series) -> pd.Series:
    """Helper: parsea columna de strings que son listas."""
    return series.apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])


def load_and_parse_tracks(path: Path) -> pd.DataFrame:
    """Carga y parsea tracks."""
    df = pd.read_csv(path)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['artists'] = _parse_list_column(df['artists'])
    df['id_artists'] = _parse_list_column(df['id_artists']) 
    df['n_artists'] = df['artists'].apply(len)
    return df

def load_and_parse_artists(path: Path) -> pd.DataFrame:
    """Carga y parsea artists."""
    df = pd.read_csv(path)
    df['genres'] = _parse_list_column(df['genres'])
    return df

def aggregate_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de audio + año de debut + n_tracks por artista."""
    df = (tracks_df
        .explode('id_artists')
        .rename(columns={'id_artists': 'artist_id'}))
    
    agg_dict = {feature: 'mean' for feature in AUDIO_FEATURES}
    agg_dict['release_year'] = 'min'
    agg_dict['id'] = 'count'  # Cuenta tracks por artista
    
    result = df.groupby('artist_id').agg(agg_dict).reset_index()
    result.rename(columns={'id': 'n_tracks'}, inplace=True)  
    
    return result
    
def get_top_genres(artists_df: pd.DataFrame, top_n: int) -> list:
    """Obtiene los top-N géneros más comunes."""
    return artists_df['genres'].explode().value_counts().head(top_n).index.tolist()

def create_genre_one_hot(artists_df: pd.DataFrame, top_genres: list) -> pd.DataFrame:
    """Crea columnas one-hot para los géneros top-N."""
    df = artists_df[['id', 'genres']].copy()  # Solo lo necesario
    
    for genre in top_genres:
        df[f'genre_{genre}'] = df['genres'].apply(lambda g: int(genre in g))
    
    # Devuelve solo one-hot, con 'id' como índice
    return df.set_index('id')[[f'genre_{g}' for g in top_genres]]

def merge_features(
    artist_audio: pd.DataFrame, 
    genre_onehot: pd.DataFrame, 
    artists_df: pd.DataFrame
) -> pd.DataFrame:
    """Mergea features de audio y géneros."""
    genre_reset = genre_onehot.reset_index()
    artists_reset = artists_df[['id', 'popularity', 'followers']].reset_index(drop=True)
    
    merged = (artist_audio
        .merge(genre_reset, left_on='artist_id', right_on='id', how='inner', suffixes=('', '_y'))
        .merge(artists_reset, left_on='artist_id', right_on='id', how='inner', suffixes=('', '_y')))
    
    # Limpia columnas innecesarias (AÑADE 'n_tracks')
    cols_to_keep = ['artist_id', 'n_tracks'] + AUDIO_FEATURES + [c for c in merged.columns if c.startswith('genre_')] + ['popularity', 'followers', 'release_year']
    return merged[cols_to_keep]

def add_decade_and_filter(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega decade y filtra según 4 criterios."""
    df = merged_df.copy()
    
    # Crear decade
    df['decade'] = (df['release_year'] // 10) * 10
    
    # Contar géneros asignados
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    df['n_genres'] = df[genre_cols].sum(axis=1)
    
    # Filtrado: intersección de 4 criterios
    filtered_df = df[
        (df['n_genres'] >= 1) &  # Género asignado
        (df['n_tracks'] >= MIN_SONGS_PER_ARTIST) &  # Mínimo canciones
        (df['popularity'] >= MIN_POPULARITY) &  # Popularidad
        (df['followers'] >= MIN_FOLLOWERS) &  # Followers
        (df['release_year'].notnull())  # Año disponible
    ]
    
    return filtered_df

def normalize_and_save(filtered_df: pd.DataFrame, output_dir: Path) -> None:
    """Normaliza features de audio y guarda CSV."""
    df = filtered_df.copy()
    
    # Columnas a normalizar
    cols_to_normalize = AUDIO_FEATURES + ['decade']
    
    # Normalizar con StandardScaler
    scaler = StandardScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    # Seleccionar columnas finales: artist_id + audio normalizados + genres + metadatos
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    cols_final = ['artist_id'] + cols_to_normalize + genre_cols + ['popularity', 'followers', 'release_year']
    df_final = df[cols_final]
    
    # Guardar CSV
    output_path = output_dir / "features_por_artista.csv"
    df_final.to_csv(output_path, index=False)
    
    # Guardar scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✓ Features normalizadas guardadas en: {output_path}")
    print(f"✓ Scaler guardado en: {scaler_path}")
    
    
def main() -> None:
    ensure_directories()
    
    print("=" * 60)
    print("BLOQUE 3: PREPROCESADO")
    print("=" * 60)
    
    print("\n[1] Cargando y parseando datos...")
    tracks_df = load_and_parse_tracks(RAW_DATA_DIR / "tracks.csv")
    artists_df = load_and_parse_artists(RAW_DATA_DIR / "artists.csv")
    
    print(f"Tracks: {len(tracks_df):,}")
    print(f"Artists: {len(artists_df):,}")
    
    print("\n[2] Agregando features de audio...")
    artist_audio = aggregate_audio_features(tracks_df)
    
    print(f"Artistas únicos (con features): {len(artist_audio):,}")
    print(f"\nPrimeras 3 filas:\n{artist_audio.head(3)}")
    print(f"\nNaNs por columna:\n{artist_audio.isnull().sum()}")
    print(f"\nEstadísticos básicos:\n{artist_audio[AUDIO_FEATURES].describe()}")
    
    print("\n[3] Extrayendo top-30 géneros...")
    top_genres = get_top_genres(artists_df, TOP_N_GENRES)
    print(f"Top 30 géneros: {top_genres}")
    print(f"Total: {len(top_genres)}")
    
    print("\n[4] Creando one-hot de géneros...")
    genre_onehot = create_genre_one_hot(artists_df, top_genres)
    print(f"Shape: {genre_onehot.shape}")
    print(f"Primeras 3 filas:\n{genre_onehot.head(3)}")
    
    print(f"\nÍndice de genre_onehot: {genre_onehot.index.name}")
    print(f"Primeras 3 índices: {genre_onehot.index[:3].tolist()}")
    print(f"\n¿Cuántos artistas con al menos 1 género?\n{(genre_onehot.sum(axis=1) > 0).sum():,}")
    
    print("\n[5] Mergeando features...")
    merged = merge_features(artist_audio, genre_onehot, artists_df)
    print(f"Shape después de merge: {merged.shape}")
    print(f"Primeras 3 filas:\n{merged.head(3)}")
    print(f"\nColumnas: {merged.columns.tolist()}")
    
    print("\n[6] Creando decade y filtrando...")
    filtered_df = add_decade_and_filter(merged)
    print(f"Artistas antes de filtro: {len(merged):,}")
    print(f"Artistas después de filtro: {len(filtered_df):,}")
    print(f"\nDistribución por década:\n{filtered_df['decade'].value_counts().sort_index()}")
    
    print("\n[7] Normalizando y guardando...")
    normalize_and_save(filtered_df, PROCESSED_DATA_DIR)

    print(f"\n✓ BLOQUE 3 COMPLETADO")
    print(f"Artistas finales: {len(filtered_df):,}")


if __name__ == "__main__":
    main()