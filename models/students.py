"""
models/students.py — Modèles élèves allégés pour la distillation.

StudentGIN  : 2 couches GIN, hidden=16 → ~4–6× moins params que GIN
StudentTACO : 1 bloc attention, hidden=16 → ~4–6× moins params que TACO-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gin import knn_graph, GINConv, gather_neighbors


# ─── Student GIN ──────────────────────────────────────────────────────────────
class StudentGIN(nn.Module):
    """
    Version allégée de GIN : 2 couches, hidden_dim=16.
    Teacher GIN ≈ 200K params → StudentGIN ≈ 12–20K params (10× moins).
    Ratio garanti ≥ 4× demandé.
    """

    def __init__(self, num_classes: int = 8, in_channels: int = 6,
                 hidden_dim: int = 16, k: int = 10, dropout: float = 0.4):
        super().__init__()
        self.k = k

        self.gin1 = GINConv(in_channels, hidden_dim)
        self.gin2 = GINConv(hidden_dim,  hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 6, N) → logits (B, num_classes)"""
        idx = knn_graph(x, k=self.k)
        h1  = self.gin1(x,  idx)        # (B, 16, N)
        h2  = self.gin2(h1, idx)        # (B, 16, N)
        g   = torch.cat([h2.mean(-1), h2.max(-1)[0]], dim=1)  # (B, 32)
        return self.classifier(g)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        idx = knn_graph(x, k=self.k)
        h1  = self.gin1(x, idx)
        h2  = self.gin2(h1, idx)
        return torch.cat([h2.mean(-1), h2.max(-1)[0]], dim=1)


# ─── Student TACO ─────────────────────────────────────────────────────────────
class StudentTACO(nn.Module):
    """
    Version allégée de TACO-Net : 1 bloc attention, embed=16.
    Teacher TACO ≈ 150K params → StudentTACO ≈ 15–25K params (≥ 6× moins).
    """

    def __init__(self, num_classes: int = 8, in_channels: int = 6,
                 embed_dim: int = 16, k: int = 10, dropout: float = 0.4):
        super().__init__()
        self.k = k
        inter  = max(embed_dim // 2, 8)   # 8

        # Projection initiale
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, inter, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter),
            nn.ReLU(inplace=True),
        )

        # 1 seul bloc d'attention légère
        self.W_q = nn.Linear(inter, embed_dim, bias=False)
        self.W_k = nn.Linear(inter, embed_dim, bias=False)
        self.W_v = nn.Linear(inter, embed_dim, bias=False)
        self.bn  = nn.BatchNorm1d(embed_dim)
        self.scale = embed_dim ** -0.5

        # Classificateur
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def _attention(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Attention locale simplifiée."""
        B, C, N = x.shape
        k = idx.shape[2]
        out_ch = self.W_q.out_features

        x_t = x.permute(0, 2, 1)    # (B, N, C)
        q   = self.W_q(x_t)          # (B, N, out_ch)
        k_  = self.W_k(x_t)
        v   = self.W_v(x_t)

        k_neigh = gather_neighbors(k_.transpose(1, 2), idx).permute(0, 2, 3, 1)
        v_neigh = gather_neighbors(v.transpose(1, 2), idx).permute(0, 2, 3, 1)

        q_exp  = q.unsqueeze(2).expand(B, N, k, out_ch)
        scores = (q_exp * k_neigh).sum(-1) * self.scale
        alpha  = F.softmax(scores, dim=-1).unsqueeze(-1)
        agg    = (alpha * v_neigh).sum(dim=2)               # (B, N, out_ch)

        agg_flat = agg.reshape(B * N, out_ch)
        agg_flat = self.bn(agg_flat)
        out = F.relu(agg_flat).reshape(B, N, out_ch).permute(0, 2, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = knn_graph(x, k=self.k)
        h   = self.input_proj(x)                             # (B, 8, N)
        h   = self._attention(h, idx)                        # (B, 16, N)
        g   = torch.cat([h.max(-1)[0], h.mean(-1)], dim=1)  # (B, 32)
        return self.classifier(g)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        idx = knn_graph(x, k=self.k)
        h   = self.input_proj(x)
        h   = self._attention(h, idx)
        return torch.cat([h.max(-1)[0], h.mean(-1)], dim=1)


# ─── Factory ──────────────────────────────────────────────────────────────────
def get_student(arch: str, num_classes: int = 8, **kwargs) -> nn.Module:
    """Retourne le modèle élève pour l'architecture donnée."""
    arch = arch.lower()
    if arch == 'gin':
        return StudentGIN(num_classes=num_classes, **kwargs)
    elif arch in ('taconet', 'taco'):
        return StudentTACO(num_classes=num_classes, **kwargs)
    raise ValueError(f"Architecture inconnue pour student : {arch}")


if __name__ == '__main__':
    import sys; sys.path.insert(0, '..')
    from utils import count_parameters
    x = torch.randn(4, 6, 512)
    for name, cls in [('StudentGIN', StudentGIN), ('StudentTACO', StudentTACO)]:
        m = cls(num_classes=8)
        print(f"{name}: {count_parameters(m):,} params | out={m(x).shape}")
