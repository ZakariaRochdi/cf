"""
federated/hfl.py — Horizontal Federated Learning (FedAvg manuel).

Implémentation MANUELLE de FedAvg sans bibliothèque tierce.
Utilise uniquement state_dict et copy.deepcopy.

Protocole FedAvg :
  1. Serveur diffuse le modèle global via copy.deepcopy
  2. Chaque client s'entraîne localement (2 epochs)
  3. Les clients renvoient leur state_dict + taille du dataset
  4. Serveur agrège via FedAvg (moyenne pondérée par n_k)

Formule : w^(t+1) = Σ_k (n_k / n) · w_k^(t)
"""

import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Any


# ─── FedAvg : Agrégation manuelle ─────────────────────────────────────────────
def fedavg_aggregate(global_model: nn.Module,
                     client_updates: List[Tuple[Dict[str, torch.Tensor], int]]
                     ) -> nn.Module:
    """
    Agrégation FedAvg : moyenne pondérée des state_dict par taille de dataset.

    Args:
        global_model   : modèle global à mettre à jour
        client_updates : liste de (state_dict, n_samples)
    Returns:
        global_model mis à jour in-place
    """
    if not client_updates:
        return global_model

    total_samples = sum(n for _, n in client_updates)
    global_sd     = global_model.state_dict()
    agg_sd        = {}

    for key in global_sd.keys():
        ref = global_sd[key]
        if ref.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            # Moyenne pondérée des paramètres flottants
            weighted = torch.zeros_like(ref, dtype=torch.float64)
            for sd, n_k in client_updates:
                if key in sd:
                    weighted += sd[key].double() * (n_k / total_samples)
            agg_sd[key] = weighted.to(ref.dtype)
        else:
            # Buffers entiers (ex: BatchNorm.num_batches_tracked) → max
            vals = [sd[key] for sd, _ in client_updates if key in sd]
            agg_sd[key] = torch.stack(vals).max(dim=0)[0] if vals else ref

    global_model.load_state_dict(agg_sd)
    return global_model


# ─── Entraînement local d'un client ───────────────────────────────────────────
def client_train(model: nn.Module, loader: DataLoader,
                 lr: float, local_epochs: int,
                 device: torch.device) -> Tuple[Dict, int, float]:
    """
    Entraîne le modèle local sur les données privées du client.

    Returns:
        (state_dict mis à jour, n_samples, loss_moyen)
    """
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    n_samples = len(loader.dataset)
    total_loss = 0.0
    total      = 0

    for _ in range(local_epochs):
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pts)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total      += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    return copy.deepcopy(model.state_dict()), n_samples, avg_loss


# ─── Évaluation du modèle global ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_global(model: nn.Module, loader: DataLoader,
                    device: torch.device) -> Tuple[float, float]:
    """Évalue le modèle global sur le jeu de validation."""
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        logits = model(pts)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


# ─── Boucle principale HFL ────────────────────────────────────────────────────
class HorizontalFL:
    """
    Orchestrateur de Federated Learning Horizontal.
    Simule Serveur + N Clients sur la même machine CPU.
    """

    def __init__(self,
                 global_model:    nn.Module,
                 client_loaders:  List[DataLoader],
                 val_loader:      DataLoader,
                 n_rounds:        int   = 5,
                 local_epochs:    int   = 2,
                 lr:              float = 1e-3,
                 device:          torch.device = torch.device('cpu'),
                 client_names:    List[str]    = None):
        self.global_model   = global_model.to(device)
        self.client_loaders = client_loaders
        self.val_loader     = val_loader
        self.n_rounds       = n_rounds
        self.local_epochs   = local_epochs
        self.lr             = lr
        self.device         = device
        self.n_clients      = len(client_loaders)
        self.client_names   = client_names or [f'Client {i}' for i in range(self.n_clients)]
        self.history        = {
            'round': [], 'val_acc': [], 'val_loss': [],
            'client_losses': [[] for _ in range(self.n_clients)]
        }

    def run(self) -> Dict[str, Any]:
        """Lance la fédération horizontale et retourne l'historique."""
        print(f"\n{'='*60}")
        print(f"  HFL — {self.n_clients} clients | "
              f"{self.n_rounds} rounds | {self.local_epochs} epochs locales")
        for i, (name, loader) in enumerate(zip(self.client_names, self.client_loaders)):
            print(f"  • {name}: {len(loader.dataset)} exemples")
        print(f"{'='*60}")

        t_start = time.time()

        for rnd in range(1, self.n_rounds + 1):
            print(f"\n  ─── Round {rnd}/{self.n_rounds} ───")
            client_updates = []

            for cid, loader in enumerate(self.client_loaders):
                if len(loader.dataset) == 0:
                    print(f"  [{self.client_names[cid]}] dataset vide, ignoré.")
                    continue

                # ① Serveur → Client : copie du modèle global
                local_model = copy.deepcopy(self.global_model)

                # ② Client : entraînement local
                t0 = time.time()
                sd, n_k, loss_k = client_train(
                    local_model, loader, self.lr, self.local_epochs, self.device)
                elapsed = time.time() - t0

                client_updates.append((sd, n_k))
                self.history['client_losses'][cid].append(loss_k)
                print(f"  [{self.client_names[cid]}] "
                      f"{n_k} exemples | Loss={loss_k:.4f} | {elapsed:.1f}s")

            if not client_updates:
                print(f"  Round {rnd}: aucune mise à jour.")
                continue

            # ③ Serveur : agrégation FedAvg
            self.global_model = fedavg_aggregate(self.global_model, client_updates)

            # ④ Évaluation globale
            val_loss, val_acc = evaluate_global(
                self.global_model, self.val_loader, self.device)
            self.history['round'].append(rnd)
            self.history['val_acc'].append(val_acc)
            self.history['val_loss'].append(val_loss)

            print(f"  [Global] Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        total_time = time.time() - t_start
        self.history['total_time'] = total_time
        print(f"\n  ✓ Temps total : {total_time/60:.1f} min")
        return self.history

    def get_per_client_accuracy(self) -> Dict[str, float]:
        """Calcule l'accuracy du modèle global sur chaque client."""
        per_client = {}
        for cid, loader in enumerate(self.client_loaders):
            if len(loader.dataset) == 0:
                per_client[self.client_names[cid]] = 0.0
                continue
            _, acc = evaluate_global(self.global_model, loader, self.device)
            per_client[self.client_names[cid]] = acc
        return per_client
