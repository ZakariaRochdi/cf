"""
train_centralized.py — Partie 1 : Baseline Centralisée.

Entraîne GIN et TACO-Net sur le même sous-ensemble de données.
Génère les 4 premières courbes obligatoires (Loss/Accuracy × 2 modèles).

Usage :
    python train_centralized.py [--epochs N] [--batch B] [--max_cls N] [--quick]
"""

import os
import sys
import time
import json
import copy
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from dataset  import get_dataloaders, NUM_CLASSES, SELECTED_CLASSES
from models   import get_model
from utils    import (EarlyStopping, count_parameters, evaluate_model,
                      plot_gin_curves, plot_taconet_curves,
                      plot_accuracy_bar_chart, plot_params_bar_chart,
                      plot_training_time_bar, plot_confusion_matrix,
                      print_summary_table, save_results_json, REPORTS_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, 'labeled_dataset.csv')
DEVICE   = torch.device('cpu')


def parse_args():
    p = argparse.ArgumentParser(description='Baseline Centralisée — GIN & TACO-Net')
    p.add_argument('--epochs',   type=int,   default=30,
                   help='Nombre max d\'époques (défaut: 30)')
    p.add_argument('--batch',    type=int,   default=8,
                   help='Batch size (défaut: 8)')
    p.add_argument('--lr',       type=float, default=1e-3,
                   help='Learning rate (défaut: 0.001)')
    p.add_argument('--max_cls',  type=int,   default=500,
                   help='Max objets par classe (défaut: 500)')
    p.add_argument('--patience', type=int,   default=5,
                   help='Patience early stopping (défaut: 5)')
    p.add_argument('--k',        type=int,   default=10,
                   help='Voisins k-NN (défaut: 10)')
    p.add_argument('--quick',    action='store_true',
                   help='Mode rapide : 3 epochs, 50 obj/classe (test pipeline)')
    return p.parse_args()


# ─── Boucles d'entraînement ───────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(pts)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        logits = model(pts)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def train_model(arch: str, train_loader, val_loader,
                n_epochs: int, lr: float, patience: int,
                k: int = 10) -> tuple:
    """
    Entraîne un modèle (arch='gin' ou 'taconet') avec early stopping.
    Retourne (best_model, history, train_time_seconds).
    """
    model     = get_model(arch, num_classes=NUM_CLASSES, k=k).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    stopper   = EarlyStopping(patience=patience)

    n_params = count_parameters(model)
    arch_name = arch.upper() if arch == 'gin' else 'TACO-Net'
    print(f"\n{'='*60}")
    print(f"  Baseline — {arch_name}")
    print(f"  Paramètres : {n_params:,}")
    print(f"  Device     : {DEVICE} | LR={lr} | Epochs max={n_epochs} | k={k}")
    print(f"{'='*60}")

    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    t_start = time.time()
    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
        ep_time = time.time() - t0

        history['epoch'].append(epoch)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        epoch_times.append(ep_time)

        print(f"  Epoch {epoch:3d}/{n_epochs} | "
              f"Train Loss={tr_loss:.4f} Acc={tr_acc:.3f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.3f} | "
              f"{ep_time:.1f}s")

        if stopper(val_acc, model):
            print(f"  [EarlyStopping] Arrêt à l'époque {epoch} "
                  f"(patience={patience} dépassée)")
            break

    total_time = time.time() - t_start
    avg_ep_time = sum(epoch_times) / len(epoch_times)
    best_model = stopper.load_best(model)

    print(f"\n  ✓ Meilleure Val Acc = {stopper.best_score:.4f}")
    print(f"  ✓ Temps total       = {total_time/60:.1f} min "
          f"({avg_ep_time:.1f}s/époque)")

    return best_model, history, total_time


# ─── Point d'entrée ───────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.quick:
        args.epochs   = 3
        args.max_cls  = 50
        args.patience = 2
        print("\n  [QUICK] Mode rapide (3 epochs, 50 obj/classe)\n")

    print(f"\n{'#'*60}")
    print(f"  PARTIE 1 — Baseline Centralisée")
    print(f"  Dataset  : {CSV_PATH}")
    print(f"  Classes  : {NUM_CLASSES} ({', '.join(SELECTED_CLASSES)})")
    print(f"  Epochs   : max {args.epochs} | Batch: {args.batch}")
    print(f"  Device   : {DEVICE}")
    print(f"{'#'*60}\n")

    # ── Chargement des données ────────────────────────────────────────────────
    print("Chargement du dataset...")
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        CSV_PATH, max_per_class=args.max_cls, batch_size=args.batch)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    all_results    = {}
    all_params     = {}
    all_times      = {}
    all_histories  = {}
    all_preds_data = {}

    # ── Entraînement des deux modèles ─────────────────────────────────────────
    for arch in ['gin', 'taconet']:
        arch_label = 'GIN' if arch == 'gin' else 'TACO-Net'

        model, history, train_time = train_model(
            arch, train_loader, val_loader,
            n_epochs=args.epochs, lr=args.lr,
            patience=args.patience, k=args.k)

        # Évaluation sur test
        print(f"\n  Évaluation {arch_label} sur test set...")
        test_metrics = evaluate_model(model, test_loader, DEVICE, SELECTED_CLASSES)
        test_metrics['train_time'] = train_time
        test_metrics['n_params']   = count_parameters(model)

        all_results[f'{arch_label} Baseline']  = test_metrics
        all_params[arch_label]                 = count_parameters(model)
        all_times[arch_label]                  = train_time
        all_histories[arch]                    = history
        all_preds_data[arch]                   = {}

        # Sauvegarde du checkpoint
        ckpt_path = os.path.join(REPORTS_DIR, f'best_{arch}.pt')
        torch.save(model.state_dict(), ckpt_path)
        # Sauvegarde de l'historique
        hist_path = os.path.join(REPORTS_DIR, f'history_{arch}.json')
        with open(hist_path, 'w') as f:
            json.dump(history, f)
        print(f"  [OK] Checkpoint → {ckpt_path}")

        # Matrice de confusion
        @torch.no_grad()
        def get_preds(m, loader):
            m.eval()
            preds, labs = [], []
            for pts, labels in loader:
                logits = m(pts)
                preds.extend(logits.argmax(1).numpy().tolist())
                labs.extend(labels.numpy().tolist())
            return preds, labs

        preds_test, labels_test = get_preds(model, test_loader)
        plot_confusion_matrix(preds_test, labels_test,
                              SELECTED_CLASSES, f'{arch_label} Baseline')

    # ── Génération des graphiques ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Génération des graphiques...")

    # Graphiques 1 & 2 : GIN
    plot_gin_curves(all_histories['gin'])

    # Graphiques 3 & 4 : TACO-Net
    plot_taconet_curves(all_histories['taconet'])

    # Graphique 7 : Accuracy bar (baseline seulement ici, mis à jour après)
    plot_accuracy_bar_chart({k: v['accuracy'] for k, v in all_results.items()})

    # Graphique 8 : Parameter count
    plot_params_bar_chart(all_params)

    # Graphique 9 : Training time
    plot_training_time_bar(all_times)

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Sauvegarde JSON ───────────────────────────────────────────────────────
    save_results_json(all_results, 'centralized_results.json')

    print(f"\n{'#'*60}")
    print(f"  ✅ Partie 1 terminée. Graphiques → plots/")
    print(f"  ✅ Résultats      → reports/")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
