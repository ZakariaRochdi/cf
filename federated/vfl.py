"""
federated/vfl.py — Vertical Federated Learning (VFL).

Partition par attributs : Entité A (XYZ) et Entité B (RGB).
Version CPU-optimisée avec encodeurs légers (embed_dim=64).

Protocole VFL :
  1. Entité A encode XYZ → embedding e_A ∈ R^64
  2. Entité B encode RGB → embedding e_B ∈ R^64
  3. Serveur : concat [e_A || e_B] → FC → prédiction
  4. Rétropropagation via autograd (gradients partagés via les embeddings)

Garantie de confidentialité : aucune entité ne voit les données brutes de l'autre.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any


# ─── Encodeur XYZ (Entité A — Géométrie) ─────────────────────────────────────
class XYZEncoder(nn.Module):
    """
    Entité A : traite uniquement les coordonnées spatiales (x, y, z).
    Mini PointNet-like : MLP partagé + max pooling global.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3,  32, 1, bias=False), nn.BatchNorm1d(32),  nn.ReLU(True),
            nn.Conv1d(32, 64, 1, bias=False), nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Conv1d(64, embed_dim, 1, bias=False),
            nn.BatchNorm1d(embed_dim), nn.ReLU(True),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Args: xyz (B, 3, N) → e_A (B, embed_dim)"""
        return self.net(xyz).max(dim=-1)[0]


# ─── Encodeur RGB (Entité B — Apparence) ─────────────────────────────────────
class RGBEncoder(nn.Module):
    """
    Entité B : traite uniquement les valeurs de couleur (r, g, b).
    Architecture symétrique à XYZEncoder.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3,  32, 1, bias=False), nn.BatchNorm1d(32),  nn.ReLU(True),
            nn.Conv1d(32, 64, 1, bias=False), nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Conv1d(64, embed_dim, 1, bias=False),
            nn.BatchNorm1d(embed_dim), nn.ReLU(True),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Args: rgb (B, 3, N) → e_B (B, embed_dim)"""
        return self.net(rgb).max(dim=-1)[0]


# ─── Serveur de Fusion ────────────────────────────────────────────────────────
class ServerFusion(nn.Module):
    """
    Serveur central : reçoit [e_A || e_B] et produit les logits.
    Ne voit jamais les données brutes XYZ ou RGB.
    """
    def __init__(self, embed_dim: int = 64, num_classes: int = 8,
                 dropout: float = 0.4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, e_A: torch.Tensor, e_B: torch.Tensor) -> torch.Tensor:
        """Args: e_A (B, d), e_B (B, d) → logits (B, num_classes)"""
        return self.fc(torch.cat([e_A, e_B], dim=1))


# ─── VFL Orchestrateur ────────────────────────────────────────────────────────
class VerticalFL:
    """
    Orchestrateur de Federated Learning Vertical.
    Simule Entité A + Entité B + Serveur sur la même machine.
    """

    def __init__(self,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 num_classes:  int   = 8,
                 embed_dim:    int   = 64,
                 lr:           float = 1e-3,
                 n_epochs:     int   = 15,
                 device:       torch.device = torch.device('cpu')):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.n_epochs     = n_epochs

        # Instanciation des composants
        self.enc_A  = XYZEncoder(embed_dim).to(device)
        self.enc_B  = RGBEncoder(embed_dim).to(device)
        self.server = ServerFusion(embed_dim, num_classes).to(device)

        # Optimiseurs séparés (chaque entité gère ses propres poids)
        self.opt_A  = torch.optim.Adam(self.enc_A.parameters(),  lr=lr, weight_decay=1e-4)
        self.opt_B  = torch.optim.Adam(self.enc_B.parameters(),  lr=lr, weight_decay=1e-4)
        self.opt_sv = torch.optim.Adam(self.server.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []
        }

    def train_epoch(self) -> float:
        """Une époque d'entraînement VFL."""
        self.enc_A.train(); self.enc_B.train(); self.server.train()
        total_loss = 0.0

        for pts, labels in self.train_loader:
            pts, labels = pts.to(self.device), labels.to(self.device)
            # Séparation stricte XYZ / RGB
            xyz = pts[:, :3, :]   # Entité A : géométrie uniquement
            rgb = pts[:, 3:, :]   # Entité B : apparence uniquement

            self.opt_A.zero_grad()
            self.opt_B.zero_grad()
            self.opt_sv.zero_grad()

            # ① Encodage local (chaque entité sur ses propres données)
            e_A = self.enc_A(xyz)       # (B, embed_dim) — seul l'embedding circule
            e_B = self.enc_B(rgb)

            # ② Serveur : fusion + classification
            logits = self.server(e_A, e_B)
            loss   = self.criterion(logits, labels)

            # ③ Rétropropagation : les gradients transitent vers chaque entité
            loss.backward()
            self.opt_A.step()
            self.opt_B.step()
            self.opt_sv.step()

            total_loss += loss.item() * labels.size(0)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Évaluation sur le jeu de validation."""
        self.enc_A.eval(); self.enc_B.eval(); self.server.eval()
        total_loss, correct, total = 0.0, 0, 0

        for pts, labels in self.val_loader:
            pts, labels = pts.to(self.device), labels.to(self.device)
            xyz = pts[:, :3, :]
            rgb = pts[:, 3:, :]
            e_A    = self.enc_A(xyz)
            e_B    = self.enc_B(rgb)
            logits = self.server(e_A, e_B)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        return total_loss / total, correct / total

    def run(self) -> Dict[str, Any]:
        """Lance l'entraînement VFL complet."""
        print(f"\n{'='*60}")
        print(f"  VFL — Entité A (XYZ) + Entité B (RGB)")
        print(f"  Epochs: {self.n_epochs} | Device: {self.device}")
        print(f"{'='*60}")

        t_start = time.time()

        for epoch in range(1, self.n_epochs + 1):
            tr_loss          = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if epoch % 3 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.n_epochs} | "
                      f"Train Loss={tr_loss:.4f} | "
                      f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        total_time = time.time() - t_start
        self.history['total_time'] = total_time
        self.history['final_val_acc'] = val_acc

        print(f"\n  ✓ VFL terminé. Val Acc finale = {val_acc:.4f}")
        print(f"  ✓ Temps total : {total_time/60:.1f} min")
        return self.history
