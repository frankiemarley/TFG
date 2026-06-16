"""
Bloque 6: Evaluación con MusicBrainz ground truth.
"""

import musicbrainzngs as mb
import numpy as np
import pandas as pd
import faiss
from sklearn.metrics import roc_auc_score, precision_recall_curve

from paths import PROCESSED_DATA_DIR

# Configurar user agent (requerido por MusicBrainz)
mb.set_useragent("MyApp/1.0 (contact@email.com)")

def load_faiss_data():
    """Carga índice FAISS y datos guardados."""
    index = faiss.read_index(str(PROCESSED_DATA_DIR / "faiss_index.index"))
    embeddings = np.load(str(PROCESSED_DATA_DIR / "embeddings.npy"))
    artist_ids = np.load(str(PROCESSED_DATA_DIR / "artist_ids.npy"), 
                        allow_pickle=True)
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_por_artista.csv")
    return index, embeddings, artist_ids, df


def get_musicbrainz_influences(mbid: str) -> set:
    """
    Obtiene artistas que influyeron en el dado.
    
    Args:
        mbid: MusicBrainz ID del artista
    
    Returns:
        set de nombres de artistas que lo influyeron
    
    TODO:
    1. mb.search_artists(query, limit=1) para obtener MBID si no lo tienes
    2. mb.get_artist_by_id(mbid, includes=['relationships'])
    3. Buscar en relationships aquellas con type='influenced_by'
    4. Extraer artistas y retornar como set
    5. Manejar excepciones: artista no existe, sin influencias, etc.
    
    Hint: mb.get_artist_by_id() retorna dict con estructura:
    {
      'artist': {
        'relationships': [
          {'type': 'influenced_by', 'target-type': 'artist', 'artist': {...}}
        ]
      }
    }
    """
    pass


def evaluate_model(index, embeddings, artist_ids, df, k=10, sample_size=100):
    """
    Evalúa el modelo comparando predicciones vs influencias reales.
    
    Args:
        index: FAISS index
        embeddings: (N, 64) array
        artist_ids: array de Spotify IDs
        df: DataFrame con metadata
        k: top-K similares a evaluar
        sample_size: cuántos artistas chequear
    
    Returns:
        dict con métricas: recall@k, precision@k, etc.
    
    TODO:
    1. Para sample_size artistas aleatorios:
       a. Obtener influences reales de MusicBrainz
       b. Obtener top-K similares del modelo (FAISS)
       c. Chequear overlap
       d. Calcular recall, precision, MRR
    2. Promediar métricas
    3. Retornar dict
    
    Pseudocódigo:
    recalls, precisions, mrrs = [], [], []
    for artist_id in sample(artist_ids, sample_size):
        real_influences = get_musicbrainz_influences(mbid)
        if len(real_influences) == 0:
            continue  # Sin ground truth
        
        top_k_idx = ... # FAISS search
        top_k_names = [artist_ids[i] for i in top_k_idx]
        
        overlap = len(set(top_k_names) & real_influences)
        recall = overlap / len(real_influences)
        precision = overlap / k if overlap > 0 else 0
        # MRR = 1 / (posición de primer match + 1), 0 si no hay match
        
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
    
    return {
        'recall@k': np.mean(recalls),
        'precision@k': np.mean(precisions),
        'mrr': np.mean(mrrs),
        'sample_size': len(recalls)
    }
    """
    pass


def main():
    print("=" * 60)
    print("BLOQUE 6: EVALUACIÓN (MusicBrainz)")
    print("=" * 60)
    
    # [1] Cargar datos
    print("\n[1] Cargando FAISS...")
    index, embeddings, artist_ids, df = load_faiss_data()
    print(f"Artistas: {len(artist_ids)}")
    
    # [2] Mapeo Spotify → MusicBrainz
    print("\n[2] Mapeo Spotify ID → MusicBrainz ID...")
    # TODO: crear diccionario {spotify_id: mbid}
    # Opción simple: usar columna 'name' para buscar en MB
    
    # [3] Evaluar
    print("\n[3] Evaluando modelo...")
    metrics = evaluate_model(index, embeddings, artist_ids, df, k=10, sample_size=50)
    
    print(f"\nResultados:")
    print(f"  Recall@10: {metrics['recall@k']:.4f}")
    print(f"  Precision@10: {metrics['precision@k']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")
    print(f"  Artistas evaluados: {metrics['sample_size']}")
    
    print("\n✓ BLOQUE 6 COMPLETADO")


if __name__ == "__main__":
    main()