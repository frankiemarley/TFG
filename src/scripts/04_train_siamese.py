"""
Bloque 4: Entrenamiento de Red Siamesa.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

from models.siamese import ArtistEncoder, SiameseNet, TripletLoss
from paths import PROCESSED_DATA_DIR


def main():
    print("Bloque 4: Siamese Network")
    
    # [1] Cargar datos
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_por_artista.csv")
    print(f"Datos cargados: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")
    
    # [2] Selecciona solo features (excluye metadata)
    # Mantén: audio_features + géneros + decade
    # Excluye: artist_id, popularity, followers, release_year
    exclude_cols = ['artist_id', 'popularity', 'followers', 'release_year']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    print(f"Features shape: {X.shape}")
    
    # [3] Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # [4] Modelo
    model = SiameseNet(input_dim=X.shape[1], embedding_dim=64).to(device)
    
    # [5] Loss y optimizer
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n✓ Setup completado. Listo para entrenar.")

if __name__ == "__main__":
    main()