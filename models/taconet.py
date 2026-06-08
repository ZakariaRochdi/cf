"""
models/taconet.py — TACO-Net (Token-based Attention with Compact Operations).
Version simplifiée CPU pour la classification de nuages de points 3D.

Architecture :
  - Projection initiale : 6 → 32
  - 2 blocs d'attention locale (k=10 voisins, 1 tête)
  - Agrégation globale : Max + Mean Pool → 128
  - FC classifier : 128 → 64 → num_classes
  - Embedding dim : 64 | Params : ~150K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gin import knn_graph, gather_neighbors


# ─── Bloc d'Attention Locale ──────────────────────────────────────────────────
class LocalAttentionBlock(nn.Module):
    """
    Bloc d'attention locale sur les voisins k-NN.
    Pour chaque point, calcule une attention scalaire sur ses k voisins,
    puis agrège les features pondérées.

    Analogie : Transformer local sur le nuage de points.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_ch  = in_channels
        self.out_ch = out_channels

        # Projections Q, K, V
        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)

        # Normalisation post-attention
        self.bn  = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        # Scale pour l'attention
        self.scale = out_channels ** -0.5

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, C, N)
            idx : (B, N, k)
        Returns:
            out : (B, out_channels, N)
        """
        B, C, N = x.shape
        k       = idx.shape[2]
        out_ch  = self.out_ch

        # Transposer pour les linéaires : (B, N, C)
        x_t = x.permute(0, 2, 1)   # (B, N, C)

        # Projections
        q = self.W_q(x_t)          # (B, N, out_ch)
        k_ = self.W_k(x_t)         # (B, N, out_ch)
        v  = self.W_v(x_t)         # (B, N, out_ch)

        # Gather des K/V voisins
        # idx : (B, N, k) → gather dans (B, N, out_ch)
        k_neigh = gather_neighbors(k_.transpose(1, 2), idx).permute(0, 2, 3, 1)
        v_neigh = gather_neighbors(v.transpose(1, 2), idx).permute(0, 2, 3, 1)

        # Scores d'attention scalaires
        q_exp  = q.unsqueeze(2).expand(B, N, k, out_ch)      # (B, N, k, out_ch)
        scores = (q_exp * k_neigh).sum(-1) * self.scale       # (B, N, k)
        alpha  = F.softmax(scores, dim=-1).unsqueeze(-1)      # (B, N, k, 1)

        # Agrégation pondérée
        agg = (alpha * v_neigh).sum(dim=2)                    # (B, N, out_ch)

        # Résidu + BN + activation
        if self.in_ch == self.out_ch:
            agg = agg + x_t
        agg_flat = agg.reshape(B * N, out_ch)
        agg_flat = self.bn(agg_flat)
        agg_flat = self.act(agg_flat)
        out = agg_flat.reshape(B, N, out_ch).permute(0, 2, 1)  # (B, out_ch, N)
        return out


# ─── TACO-Net ─────────────────────────────────────────────────────────────────
class TACONet(nn.Module):
    """
    TACO-Net simplifié pour la classification 3D (CPU-optimisé).

    Architecture :
        Input(6) → Conv1D(6→32) → BN → ReLU
        → LocalAttentionBlock(32→64) → LocalAttentionBlock(64→64)
        → GlobalMaxPool(64) + GlobalMeanPool(64) → concat(128)
        → FC(128→64) → Dropout → FC(64→num_classes)
    """

    def __init__(self, num_classes: int = 8, in_channels: int = 6,
                 embed_dim: int = 64, k: int = 10, dropout: float = 0.5):
        super().__init__()
        self.k = k
        inter  = embed_dim // 2   # 32

        # Projection initiale
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, inter, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter),
            nn.ReLU(inplace=True),
        )

        # 2 blocs d'attention locale
        self.attn1 = LocalAttentionBlock(inter,    embed_dim)   # 32→64
        self.attn2 = LocalAttentionBlock(embed_dim, embed_dim)  # 64→64

        # Classificateur
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 6, N)
        Returns:
            logits : (B, num_classes)
        """
        # Graphe k-NN (sur XYZ)
        idx = knn_graph(x, k=self.k)    # (B, N, k)

        # Projection initiale
        h = self.input_proj(x)           # (B, 32, N)

        # Blocs d'attention locale
        h = self.attn1(h, idx)           # (B, 64, N)
        h = self.attn2(h, idx)           # (B, 64, N)

        # Agrégation globale
        g_max  = h.max(dim=-1)[0]       # (B, 64)
        g_mean = h.mean(dim=-1)         # (B, 64)
        g      = torch.cat([g_max, g_mean], dim=1)  # (B, 128)

        return self.classifier(g)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne l'embedding global (pour VFL)."""
        idx = knn_graph(x, k=self.k)
        h   = self.input_proj(x)
        h   = self.attn1(h, idx)
        h   = self.attn2(h, idx)
        return torch.cat([h.max(-1)[0], h.mean(-1)], dim=1)


if __name__ == '__main__':
    import sys; sys.path.insert(0, '..')
    from utils import count_parameters
    model = TACONet(num_classes=8)
    print(f"TACONet — Paramètres : {count_parameters(model):,}")
    x = torch.randn(4, 6, 512)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
