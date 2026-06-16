"""
Bloque 5: FAISS Indexing para búsqueda de similares.
"""

import torch
import numpy as np
import pandas as pd
import faiss

from models.siamese import SiameseNet
from paths import PROCESSED_DATA_DIR


def extract_embeddings(
    model: SiameseNet,
    features: torch.Tensor,
    device: str = 'cpu',
    batch_size: int = 32
) -> np.ndarray:
    """
    Extrae embeddings para todos los artistas.
    
    Args:
        model: SiameseNet entrenado
        features: (N, 42) tensor
        device: 'cpu' o 'cuda'
        batch_size: batch size para forward
    
    Returns:
        embeddings: (N, 64) numpy array
    
    TODO:
    1. Pone el modelo en eval mode: model.eval()
    2. Loop sobre batches (sin gradientes):
       - with torch.no_grad():
       - emb = model(batch)  # forward sin backward
       - acumula embeddings
    3. Retorna como numpy array
    
    Hint: torch.no_grad() desactiva cálculo de gradientes (más rápido)
    """
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size].to(device)
            emb = model(batch)
            embeddings_list.append(emb.cpu().numpy())
    
    return np.vstack(embeddings_list)


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Crea índice FAISS para búsqueda por producto interno (cosine similarity).
    
    Args:
        embeddings: (N, 64) numpy array
    
    Returns:
        index: IndexFlatIP
    
    TODO:
    1. Crea índice: index = faiss.IndexFlatIP(embedding_dim)
    2. Añade embeddings: index.add(embeddings)
    3. Retorna index
    
    Note: IndexFlatIP = Inner Product = Cosine similarity (porque embeddings están L2 normalizados)
    """
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    return index


def search_similar_artists(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    k: int = 10
) -> tuple:
    """
    Busca top-K artistas similares.
    
    Args:
        index: FAISS index
        query_embedding: (64,) embedding del artista query
        k: número de similares a retornar
    
    Returns:
        (distances, indices): arrays de tamaño k
    
    TODO:
    1. query_embedding debe ser (1, 64) para FAISS
    2. index.search(query, k) retorna (distances, indices)
    3. Retorna ambos
    """
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return distances, indices  # ← Correcto


def main():
    print("=" * 60)
    print("BLOQUE 5: FAISS INDEXING")
    print("=" * 60)
    print("DEBUG: main() iniciado")
    
    # [1] Cargar datos
    print("\n[1] Cargando datos...")
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "features_por_artista.csv")
        print(f"DEBUG: CSV cargado, shape: {df.shape}")
    except Exception as e:
        print(f"ERROR al cargar CSV: {e}")
        return
    
    # [1] Cargar datos
    print("\n[1] Cargando datos...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_por_artista.csv")
    
    exclude_cols = ['artist_id', 'popularity', 'followers', 'release_year']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    
    artist_ids = df['artist_id'].values
    print(f"Artistas: {len(df)}, Features: {X.shape[1]}")
    
    # [2] Cargar modelo
    print("\n[2] Cargando modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNet(input_dim=X.shape[1], embedding_dim=64).to(device)
    model_path = PROCESSED_DATA_DIR / "siamese_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modelo cargado desde {model_path}")
    
    # [3] Extraer embeddings
    print("\n[3] Extrayendo embeddings...")
    embeddings = extract_embeddings(model, X, device=device)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # [4] Crear índice FAISS
    print("\n[4] Creando índice FAISS...")
    index = create_faiss_index(embeddings)
    print(f"Índice creado con {index.ntotal} artistas")
    
    # [5] Ejemplo: buscar similares
    print("\n[5] Ejemplo: Top-5 similares para primeros 3 artistas...")
    for i in range(3):
        query_emb = embeddings[i:i+1]
        distances, indices = search_similar_artists(index, query_emb, k=6)
        
        print(f"\nArtista: {artist_ids[i]}")
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if rank == 0:
                print(f"  (Self)")
            else:
                print(f"  {rank}. {artist_ids[idx]} (sim: {dist:.4f})")
    
    # [6] Guardar
    print("\n[6] Guardando...")
    faiss.write_index(index, str(PROCESSED_DATA_DIR / "faiss_index.index"))
    np.save(str(PROCESSED_DATA_DIR / "embeddings.npy"), embeddings)
    np.save(str(PROCESSED_DATA_DIR / "artist_ids.npy"), artist_ids)
    print("✓ Índice, embeddings y IDs guardados")
    
    print("\n✓ BLOQUE 5 COMPLETADO")


if __name__ == "__main__":
    main()