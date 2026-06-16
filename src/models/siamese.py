import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random

from paths import PROCESSED_DATA_DIR

class ArtistEncoder(nn.Module):
    """
    MLP: 46-d features → 64-d embedding
    
    Arquitectura:
    - Input: 46 (artist_id + audio + genres + metadata)
    - Hidden: 256, 128, 64
    - Output: embedding_dim (64)
    - Activación: ReLU
    - Normalización: L2 (embeddings unitarios)
    
    TODO:
    1. __init__: define layers
        - Linear(46 → 256) + ReLU
        - Linear(256 → 128) + ReLU
        - Linear(128 → 64) + ReLU
        - Linear(64 → embedding_dim)
    2. forward(x): pasa por las capas, L2-normaliza al final
    
    Hint: F.normalize(x, p=2, dim=1) para L2-norm
    """
        
    def __init__(self, input_dim: int = 46, embedding_dim: int = 64):
        super(ArtistEncoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: 46-d → embedding_dim con L2 normalization."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x
            
class SiameseNet(nn.Module):
    """
    Red Siamesa: dos ramas idénticas (mismo encoder).
    
    Input: (batch_size, 46) → Output: (batch_size, embedding_dim)
    
    TODO:
    1. __init__: crea un ArtistEncoder
    2. forward(anchor, positive, negative):
       - emb_anchor = encoder(anchor)
       - emb_positive = encoder(positive)
       - emb_negative = encoder(negative)
       - return (emb_anchor, emb_positive, emb_negative)
    
    También puede usarse con solo 2 inputs:
    forward(artist_a, artist_b) → (emb_a, emb_b)
    """
    
    def __init__(self, input_dim: int = 46, embedding_dim: int = 64):
        super(SiameseNet, self).__init__()
        self.encoder = ArtistEncoder(input_dim, embedding_dim)
        
    
    def forward(self, *args):
        """
        Forward pass: acepta 2 o 3 inputs.
        
        Args:
            *args: (anchor, positive, negative) o (artist_a, artist_b)
        
        Returns:
            embeddings: tuple de embeddings correspondientes
        """
        embeddings = tuple(self.encoder(x) for x in args)
        # Si recibe solo 1 input, retorna el embedding directamente
        if len(args) == 1:
            return embeddings[0]
        return embeddings
    


class TripletLoss(nn.Module):
    """
    Loss personalizado con margen.
    
    Loss = max(0, d(A, P) - d(A, N) + margen)
    
    donde d = distancia euclidiana (o cosine, según normalización)
    
    TODO:
    1. __init__(margin=1.0)
    2. forward(emb_anchor, emb_positive, emb_negative):
       - Calcula d(A, P) con pairwise_distance o euclidean_distance
       - Calcula d(A, N)
       - Loss = max(0, d_AP - d_AN + margin)
       - return loss.mean()
    
    Hint: F.pairwise_distance(x, y, p=2) calcula ||x - y||
    """
    
    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self,
        emb_anchor: torch.Tensor,
        emb_positive: torch.Tensor,
        emb_negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula el triplet loss.
        
        Args:
            emb_anchor: (batch_size, embedding_dim)
            emb_positive: (batch_size, embedding_dim)
            emb_negative: (batch_size, embedding_dim)
        
        Returns:
            loss: escalar
        """
        d_pos = (emb_anchor - emb_positive).norm(p=2, dim=1)
        d_neg = (emb_anchor - emb_negative).norm(p=2, dim=1)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        return loss.mean()  
    
def generate_triplets(
    features: torch.Tensor,
    genres: np.ndarray,  # (N, 30) matriz one-hot de géneros
    n_triplets: int = 5000
) -> list:
    """
    Genera tripletas (anchor_idx, positivo_idx, negativo_idx).
    
    Args:
        features: (N, 42) tensor de features
        genres: (N, 30) array de one-hot de géneros
        n_triplets: cuántas tripletas generar
    
    Returns:
        triplets: lista de (idx_a, idx_p, idx_n)
    
    TODO:
    1. Para cada artista i (anchor):
       a. Encuentra artistas con MISMO género (intersección de genres > 0)
       b. Encuentra artistas con DIFERENTE género (sin intersección)
    2. Para n_triplets iteraciones:
       a. Selecciona anchor aleatorio
       b. Selecciona positivo de los que comparten género
       c. Selecciona negativo de los que NO comparten género
       d. Append (idx_anchor, idx_pos, idx_neg)
    3. Retorna lista de tripletas
    
    Hint: 
    - genres[i] * genres[j] > 0 → comparten género
    - genres[i] * genres[j] == 0 → no comparten género
    """
    triplets = []
    N = features.shape[0]
    
    for _ in range(n_triplets):
        anchor_idx = random.randint(0, N - 1)
        anchor_genres = genres[anchor_idx]
        
        # Positivos: artistas que comparten al menos un género
        positive_candidates = np.where((genres @ anchor_genres) > 0)[0]
        positive_candidates = positive_candidates[positive_candidates != anchor_idx]
        
        # Negativos: artistas que no comparten ningún género
        negative_candidates = np.where((genres @ anchor_genres) == 0)[0]
        
        if len(positive_candidates) == 0 or len(negative_candidates) == 0:
            continue  # Skip if no valid positive or negative
        
        positive_idx = random.choice(positive_candidates)
        negative_idx = random.choice(negative_candidates)
        
        triplets.append((anchor_idx, positive_idx, negative_idx))
    
    
    return triplets
    
    



class TripletDataset(Dataset):
    """
    Dataset que genera tripletas on-the-fly.
    
    TODO:
    1. __init__(features, labels, triplets_per_epoch)
    2. __len__(): retorna num_triplets
    3. __getitem__(idx): retorna (feat_anchor, feat_pos, feat_neg)
    """
    
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        triplets_per_epoch: int = 5000
    ):
        self.features = features
        self.labels = labels
        self.triplets = generate_triplets(features, labels, triplets_per_epoch)
    
    def __len__(self) -> int:
        return len(self.triplets)

    
    
    def __getitem__(self, idx: int) -> tuple:
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]
        return (
            self.features[anchor_idx],
            self.features[pos_idx],
            self.features[neg_idx]
        )
        
        
def train_siamese(
    train_loader: DataLoader,
    model: SiameseNet,
    loss_fn: TripletLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    device: str = 'cpu'
) -> dict:
    """
    Entrena la red Siamesa.
    
    Args:
        train_loader: DataLoader con batches de tripletas
        model: SiameseNet
        loss_fn: TripletLoss
        optimizer: torch.optim.Adam, etc.
        epochs: número de épocas
        device: 'cpu' o 'cuda'
    
    Returns:
        dict con historia de losses
    
    TODO:
    1. Para cada epoch:
       a. Loop sobre batches de tripletas
       b. Forward: model(anchor, pos, neg) → embeddings
       c. Loss: loss_fn(emb_anchor, emb_pos, emb_neg)
       d. Backward + optimizer.step()
       e. Registra loss promedio
    2. Retorna diccionario con historia de losses
    
    Pseudocódigo:
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            anchor, pos, neg = batch
            emb_a, emb_p, emb_n = model(anchor, pos, neg)
            loss = loss_fn(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
    return {'losses': losses}
    """

    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            anchor, positive, negative = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward
            emb_anchor, emb_pos, emb_neg = model(anchor, positive, negative)
            loss = loss_fn(emb_anchor, emb_pos, emb_neg)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
    
    return {'losses': losses}

def main():
    print("=" * 60)
    print("BLOQUE 4: ENTRENAMIENTO SIAMESE NETWORK")
    print("=" * 60)
    
    # [1] Cargar datos
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_por_artista.csv")
    print(f"\n[1] Datos cargados: {df.shape}")
    
    # [2] Features y géneros
    exclude_cols = ['artist_id', 'popularity', 'followers', 'release_year']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    
    # Extrae columnas de género (genre_*)
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    genres = df[genre_cols].values  # (N, 30) numpy array
    
    print(f"Features shape: {X.shape}")
    print(f"Géneros shape: {genres.shape}")
    
    # [3] Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # [4] Crear dataset y dataloader
    print(f"\n[2] Generando tripletas...")
    dataset = TripletDataset(X, genres, triplets_per_epoch=5000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Tripletas generadas: {len(dataset)}")
    
    # [5] Modelo, loss, optimizer
    print(f"\n[3] Inicializando modelo...")
    model = SiameseNet(input_dim=X.shape[1], embedding_dim=64).to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # [6] Entrenar
    print(f"\n[4] Entrenando...")
    history = train_siamese(
        train_loader,
        model,
        criterion,
        optimizer,
        epochs=5,  # TODO: cambiar a 20 o más si tienes tiempo
        device=device
    )
    
    # [7] Guardar modelo
    print(f"\n[5] Guardando modelo...")
    model_path = PROCESSED_DATA_DIR / "siamese_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado: {model_path}")
    
    print(f"\n✓ BLOQUE 4 COMPLETADO")


if __name__ == "__main__":
    main()