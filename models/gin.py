"""
models/gin.py — Graph Isomorphism Network (GIN) pour la classification 3D.

Implémentation PyTorch pur (sans torch-geometric) optimisée CPU.
Architecture :
  - K-NN graph construction (k=10) sur coordonnées XYZ
  - 3 couches GINConv avec MLP(in→64→64)
  - Global Mean Pool + Max Pool → concat → FC classifier
  - Hidden dim : 64 | Dropout : 0.5 | Params : ~200K

Référence : Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── K-NN Graph Construction ──────────────────────────────────────────────────
def knn_graph(x: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Calcule les k plus proches voisins pour chaque point.
    Args:
        x   : (B, C, N) — features (on utilise les 3 premières dims pour la distance)
        k   : nombre de voisins
    Returns:
        idx : (B, N, k) — indices des voisins
    """
    xyz = x[:, :3, :]                           # (B, 3, N) — XYZ uniquement
    B, C, N = xyz.shape
    # Distance euclidienne via produit scalaire : D_ij = ||xi||² - 2<xi,xj> + ||xj||²
    inner = -2.0 * torch.bmm(xyz.transpose(2, 1), xyz)   # (B, N, N)
    xx    = (xyz ** 2).sum(dim=1, keepdim=True)           # (B, 1, N)
    dist  = xx.transpose(2, 1) + inner + xx               # (B, N, N)  dist²
    # Les k+1 plus petits (inclut soi-même), on exclut le premier (distance=0)
    k_eff = min(k + 1, N)
    idx   = dist.topk(k=k_eff, dim=-1, largest=False)[1]  # (B, N, k+1)
    return idx[:, :, 1:]   # (B, N, k) — on exclut soi-même


def gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Collecte les features des voisins k-NN.
    Args:
        x   : (B, C, N)
        idx : (B, N, k)
    Returns:
        neigh : (B, C, N, k)
    """
    B, C, N = x.shape
    k       = idx.shape[2]
    # Flatten idx pour gather
    idx_flat = idx.reshape(B, -1)                     # (B, N*k)
    # expand x pour gather: (B, C, N*k)
    x_t   = x.transpose(1, 2)                        # (B, N, C)
    # Gather voisins
    idx_exp = idx_flat.unsqueeze(-1).expand(B, N * k, C)   # (B, N*k, C)
    neigh   = x_t.gather(1, idx_exp)                        # (B, N*k, C)
    neigh   = neigh.reshape(B, N, k, C).permute(0, 3, 1, 2)  # (B, C, N, k)
    return neigh


# ─── GINConv Layer ────────────────────────────────────────────────────────────
class GINConv(nn.Module):
    """
    Couche GIN (Graph Isomorphism Network Convolution).
    Formule : h_v^(l+1) = MLP((1 + ε) · h_v^(l) + Σ_{u ∈ N(v)} h_u^(l))

    Implémentation avec agrégation sum des voisins k-NN.
    MLP : in → hidden → hidden (BN + ReLU)
    """

    def __init__(self, in_channels: int, hidden: int = 64,
                 eps: float = 0.0, learn_eps: bool = True):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor([eps])) if learn_eps \
                   else torch.tensor([eps])

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, C, N)
            idx : (B, N, k) indices k-NN
        Returns:
            out : (B, hidden, N)
        """
        B, C, N = x.shape
        k       = idx.shape[2]

        # Agrégation sum des voisins
        neigh     = gather_neighbors(x, idx)       # (B, C, N, k)
        agg_neigh = neigh.sum(dim=-1)               # (B, C, N) — sum pooling

        # Message GIN
        h = (1.0 + self.eps) * x + agg_neigh       # (B, C, N)

        # MLP appliqué point-par-point : reshape (B*N, C)
        h_flat = h.permute(0, 2, 1).reshape(B * N, C)
        h_flat = self.mlp(h_flat)
        out    = h_flat.reshape(B, N, -1).permute(0, 2, 1)  # (B, hidden, N)
        return out


# ─── GIN Classifier ───────────────────────────────────────────────────────────
class GIN(nn.Module):
    """
    GIN pour la classification de nuages de points 3D (CPU-optimisé).

    Architecture :
        Input(6) → GINConv(6→64) → GINConv(64→64) → GINConv(64→64)
        → GlobalMeanPool + GlobalMaxPool → FC(128→64) → FC(64→num_classes)

    Hyperparamètres CPU :
        hidden_dim = 64 | k = 10 | dropout = 0.5
    """

    def __init__(self, num_classes: int = 8, in_channels: int = 6,
                 hidden_dim: int = 64, k: int = 10, dropout: float = 0.5):
        super().__init__()
        self.k          = k
        self.num_classes = num_classes

        # 3 couches GINConv
        self.gin1 = GINConv(in_channels, hidden_dim)
        self.gin2 = GINConv(hidden_dim,  hidden_dim)
        self.gin3 = GINConv(hidden_dim,  hidden_dim)

        # Classificateur : global pool × 2 → FC
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 6, N) — nuage de points [xyz + rgb]
        Returns:
            logits : (B, num_classes)
        """
        # Construction du graphe k-NN (une seule fois, sur XYZ)
        idx = knn_graph(x, k=self.k)   # (B, N, k)

        # 3 couches GIN
        h1 = self.gin1(x,  idx)        # (B, 64, N)
        h2 = self.gin2(h1, idx)        # (B, 64, N)
        h3 = self.gin3(h2, idx)        # (B, 64, N)

        # Agrégation globale : mean + max
        g_mean = h3.mean(dim=-1)       # (B, 64)
        g_max  = h3.max(dim=-1)[0]    # (B, 64)
        g      = torch.cat([g_mean, g_max], dim=1)  # (B, 128)

        return self.classifier(g)      # (B, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne l'embedding global (pour VFL)."""
        idx = knn_graph(x, k=self.k)
        h1  = self.gin1(x,  idx)
        h2  = self.gin2(h1, idx)
        h3  = self.gin3(h2, idx)
        return torch.cat([h3.mean(-1), h3.max(-1)[0]], dim=1)


if __name__ == '__main__':
    from utils import count_parameters
    model = GIN(num_classes=8)
    print(f"GIN — Paramètres : {count_parameters(model):,}")
    x = torch.randn(4, 6, 512)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
