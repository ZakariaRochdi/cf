"""
utils.py — Utilitaires partagés : early stopping, métriques, graphiques.
Génère les 9 graphiques obligatoires du projet.
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              classification_report)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Style global des graphiques
plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 11,
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 11,
    'legend.fontsize' : 10,
    'figure.dpi'      : 120,
    'axes.grid'       : True,
    'grid.alpha'      : 0.35,
    'axes.spines.top' : False,
    'axes.spines.right': False,
})

PLOTS_DIR   = 'plots'
REPORTS_DIR = 'reports'

# Palette de couleurs cohérente
COLORS = {
    'gin'        : '#2563EB',    # bleu
    'taconet'    : '#DC2626',    # rouge
    'fed_avg'    : '#16A34A',    # vert
    'distill'    : '#9333EA',    # violet
    'student_gin': '#0891B2',    # cyan
    'student_taco': '#EA580C',   # orange
    'teacher'    : '#4B5563',    # gris
}


# ─── Early Stopping ───────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Arrêt anticipé basé sur la précision de validation.
    Sauvegarde le meilleur état du modèle (state_dict).
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.best_sd    = None
        self.stop       = False

    def __call__(self, val_acc: float, model: nn.Module) -> bool:
        import copy
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.best_sd    = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.best_sd    = copy.deepcopy(model.state_dict())
            self.counter    = 0
        return self.stop

    def load_best(self, model: nn.Module) -> nn.Module:
        """Charge les meilleurs poids dans le modèle."""
        if self.best_sd is not None:
            model.load_state_dict(self.best_sd)
        return model


# ─── Métriques ────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """Compte les paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(all_preds: list, all_labels: list,
                    class_names: list = None) -> dict:
    """
    Calcule accuracy, precision, recall, F1 (macro).
    """
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc    = accuracy_score(labels, preds)
    prec   = precision_score(labels, preds, average='macro', zero_division=0)
    rec    = recall_score(labels, preds, average='macro', zero_division=0)
    f1     = f1_score(labels, preds, average='macro', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader,
                   device: torch.device,
                   class_names: list = None) -> dict:
    """Évalue un modèle sur un DataLoader, retourne les métriques."""
    model.to(device).eval()
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0

    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        logits = model(pts)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)

    metrics = compute_metrics(all_preds, all_labels, class_names)
    metrics['loss']     = total_loss / total
    metrics['n_params'] = count_parameters(model)
    return metrics


# ─── Graphiques obligatoires ──────────────────────────────────────────────────
os.makedirs(PLOTS_DIR, exist_ok=True)


def _savefig(fig, filename: str, tight: bool = True) -> str:
    """Sauvegarde une figure dans plots/ et ferme."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    if tight:
        plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] → {path}")
    return path


# ── Graphiques 1 & 2 : GIN Loss + Accuracy ───────────────────────────────────
def plot_gin_curves(history: dict) -> tuple:
    """Graphiques 1 et 2 : Loss et Accuracy de GIN."""
    epochs = history['epoch']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('GIN — Courbes d\'Entraînement', fontweight='bold', fontsize=14)

    # Loss
    axes[0].plot(epochs, history['train_loss'], color=COLORS['gin'],
                 lw=2, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], color=COLORS['gin'],
                 lw=2, ls='--', label='Val Loss')
    axes[0].set_title('GIN — Loss'); axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Loss'); axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], color=COLORS['gin'],
                 lw=2, label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], color=COLORS['gin'],
                 lw=2, ls='--', label='Val Acc')
    axes[1].set_title('GIN — Accuracy'); axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Accuracy'); axes[1].legend()

    p1 = _savefig(fig, 'gin_loss_curve.png')

    # Courbe accuracy séparée
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(epochs, history['train_acc'], color=COLORS['gin'], lw=2, label='Train')
    ax2.plot(epochs, history['val_acc'],   color=COLORS['gin'], lw=2, ls='--', label='Val')
    ax2.set_title('GIN — Accuracy vs Époques', fontweight='bold')
    ax2.set_xlabel('Époque'); ax2.set_ylabel('Accuracy'); ax2.legend()
    p2 = _savefig(fig2, 'gin_accuracy_curve.png')
    return p1, p2


# ── Graphiques 3 & 4 : TACO-Net Loss + Accuracy ──────────────────────────────
def plot_taconet_curves(history: dict) -> tuple:
    """Graphiques 3 et 4 : Loss et Accuracy de TACO-Net."""
    epochs = history['epoch']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('TACO-Net — Courbes d\'Entraînement', fontweight='bold', fontsize=14)

    axes[0].plot(epochs, history['train_loss'], color=COLORS['taconet'],
                 lw=2, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], color=COLORS['taconet'],
                 lw=2, ls='--', label='Val Loss')
    axes[0].set_title('TACO-Net — Loss'); axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Loss'); axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], color=COLORS['taconet'],
                 lw=2, label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], color=COLORS['taconet'],
                 lw=2, ls='--', label='Val Acc')
    axes[1].set_title('TACO-Net — Accuracy'); axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Accuracy'); axes[1].legend()

    p3 = _savefig(fig, 'taconet_loss_curve.png')

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(epochs, history['train_acc'], color=COLORS['taconet'], lw=2, label='Train')
    ax2.plot(epochs, history['val_acc'],   color=COLORS['taconet'], lw=2, ls='--', label='Val')
    ax2.set_title('TACO-Net — Accuracy vs Époques', fontweight='bold')
    ax2.set_xlabel('Époque'); ax2.set_ylabel('Accuracy'); ax2.legend()
    p4 = _savefig(fig2, 'taconet_accuracy_curve.png')
    return p3, p4


# ── Graphique 5 : FedAvg Convergence ─────────────────────────────────────────
def plot_fedavg_convergence(gin_history: dict, taco_history: dict = None) -> str:
    """Graphique 5 : Convergence FedAvg (accuracy globale vs round)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('FedAvg — Convergence (Accuracy Globale)', fontweight='bold')

    if gin_history and 'round' in gin_history:
        ax.plot(gin_history['round'], gin_history['val_acc'],
                color=COLORS['gin'], lw=2.5, marker='o', ms=7, label='GIN — FedAvg')
    if taco_history and 'round' in taco_history:
        ax.plot(taco_history['round'], taco_history['val_acc'],
                color=COLORS['taconet'], lw=2.5, marker='s', ms=7, label='TACO-Net — FedAvg')

    ax.set_xlabel('Round Fédéré'); ax.set_ylabel('Val Accuracy')
    ax.set_ylim(0, 1); ax.legend()
    return _savefig(fig, 'fedavg_convergence_curve.png')


# ── Graphique 6 : Distillation Comparison ────────────────────────────────────
def plot_distillation_comparison(gin_hist: dict = None,
                                  taco_hist: dict = None) -> str:
    """Graphique 6 : Comparaison Teacher vs Student accuracy au fil des époques."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distillation — Teacher vs Student', fontweight='bold', fontsize=14)

    for ax, hist, name, color in [
        (axes[0], gin_hist,  'GIN',      COLORS['gin']),
        (axes[1], taco_hist, 'TACO-Net', COLORS['taconet']),
    ]:
        if hist:
            epochs = hist['epoch']
            ax.plot(epochs, hist['student_val_acc'], color=color,
                    lw=2, label=f'Student {name}')
            if 'teacher_acc_line' in hist:
                ax.axhline(hist['teacher_acc_line'], color=COLORS['teacher'],
                           ls='--', lw=1.5, label=f'Teacher {name}')
        ax.set_title(f'Distillation — {name}')
        ax.set_xlabel('Époque'); ax.set_ylabel('Val Accuracy')
        ax.set_ylim(0, 1); ax.legend()

    return _savefig(fig, 'distillation_comparison_curve.png')


# ── Graphique 7 : Accuracy Comparison Bar Chart ───────────────────────────────
def plot_accuracy_bar_chart(results: dict) -> str:
    """
    Graphique 7 : Histogramme des accuracy pour tous les modèles/configs.
    results = {'GIN Baseline': 0.82, 'TACO-Net Baseline': 0.79, ...}
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    names   = list(results.keys())
    values  = [v * 100 for v in results.values()]
    palette = [COLORS['gin'], COLORS['taconet'], COLORS['fed_avg'],
               COLORS['fed_avg'], COLORS['distill'], COLORS['distill'],
               COLORS['student_gin'], COLORS['student_taco']]
    palette = palette[:len(names)]

    bars = ax.bar(names, values, color=palette, alpha=0.87, edgecolor='white', lw=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Comparaison des Accuracy — Toutes Configurations',
                 fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 110)
    ax.set_xticklabels(names, rotation=30, ha='right')
    return _savefig(fig, 'accuracy_comparison_bar.png')


# ── Graphique 8 : Parameter Count Bar Chart ───────────────────────────────────
def plot_params_bar_chart(params: dict) -> str:
    """
    Graphique 8 : Histogramme du nombre de paramètres.
    params = {'GIN': 200000, 'TACO-Net': 150000, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    names   = list(params.keys())
    values  = [v / 1000 for v in params.values()]   # en milliers
    palette = [COLORS.get(n.lower().replace('-', '').replace(' ', '_'), '#64748B')
               for n in names]

    bars = ax.bar(names, values, color=palette, alpha=0.87, edgecolor='white', lw=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.0f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Nombre de Paramètres par Modèle', fontweight='bold', fontsize=14)
    ax.set_ylabel('Paramètres (×1000)')
    ax.set_xticklabels(names, rotation=20, ha='right')
    return _savefig(fig, 'parameter_count_bar.png')


# ── Graphique 9 : Training Time Bar Chart ────────────────────────────────────
def plot_training_time_bar(times: dict) -> str:
    """
    Graphique 9 : Histogramme des temps d'entraînement.
    times = {'GIN': 120.5, 'TACO-Net': 95.2, ...}  (secondes)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    names   = list(times.keys())
    values  = [v / 60 for v in times.values()]   # en minutes

    colors_list = ['#2563EB', '#DC2626', '#16A34A', '#9333EA', '#0891B2', '#EA580C']
    bars = ax.bar(names, values,
                  color=colors_list[:len(names)],
                  alpha=0.87, edgecolor='white', lw=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f} min', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Temps d\'Entraînement par Modèle', fontweight='bold', fontsize=14)
    ax.set_ylabel('Temps (minutes)')
    ax.set_xticklabels(names, rotation=20, ha='right')
    return _savefig(fig, 'training_time_bar.png')


# ── Graphique bonus : Confusion Matrix ───────────────────────────────────────
def plot_confusion_matrix(all_preds: list, all_labels: list,
                          class_names: list, model_name: str) -> str:
    cm  = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor='white',
                annot_kws={'size': 9})
    ax.set_title(f'Matrice de Confusion — {model_name}', fontweight='bold')
    ax.set_xlabel('Prédiction'); ax.set_ylabel('Vérité terrain')
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
    fname = f'confusion_{model_name.lower().replace(" ", "_")}.png'
    return _savefig(fig, fname)


# ── Tableau récapitulatif console ─────────────────────────────────────────────
def print_summary_table(results_dict: dict):
    """Affiche un tableau récapitulatif des résultats."""
    print(f"\n{'='*90}")
    print(f"  TABLEAU COMPARATIF FINAL")
    print(f"{'='*90}")
    print(f"  {'Modèle':<25} {'Acc (%)':>8} {'Precision':>10} {'Recall':>8} "
          f"{'F1':>8} {'Params':>10} {'Temps':>10}")
    print(f"  {'-'*85}")
    for name, m in results_dict.items():
        acc   = m.get('accuracy',  0) * 100
        prec  = m.get('precision', 0) * 100
        rec   = m.get('recall',    0) * 100
        f1    = m.get('f1',        0) * 100
        pars  = m.get('n_params',  0)
        sec   = m.get('train_time', 0)
        mins  = f"{sec/60:.1f}min" if sec > 0 else "—"
        print(f"  {name:<25} {acc:>8.2f} {prec:>10.2f} {rec:>8.2f} "
              f"{f1:>8.2f} {pars:>10,} {mins:>10}")
    print(f"{'='*90}\n")


def save_results_json(results_dict: dict, filename: str = 'all_results.json'):
    """Sauvegarde les résultats dans reports/."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    # Sérialiser les types numpy
    clean = {}
    for k, v in results_dict.items():
        clean[k] = {kk: float(vv) if hasattr(vv, 'item') else vv
                    for kk, vv in v.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] Résultats → {path}")
    return path


# ── Fonctions additionnelles pour main.py ─────────────────────────────────────
def plot_convergence(history: dict, arch: str, save_dir: str):
    """Génère les courbes de convergence pour une architecture."""
    epochs = history.get('epoch', [])
    if not epochs:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Loss
    axes[0].plot(epochs, history.get('train_loss', []), label='Train Loss')
    axes[0].plot(epochs, history.get('val_loss', []), label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{arch.upper()} - Loss')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(epochs, history.get('train_acc', []), label='Train Acc')
    axes[1].plot(epochs, history.get('val_acc', []), label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{arch.upper()} - Accuracy')
    axes[1].legend()
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{arch}_convergence.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_per_class(results: dict, arch: str, save_dir: str):
    """Génère un graphique par classe (placeholder)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, 'Per-class metrics\n(à implémenter)', 
            ha='center', va='center', fontsize=14)
    ax.set_title(f'{arch.upper()} - Per-Class Metrics')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{arch}_per_class.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_comparison_table(results: dict, save_dir: str):
    """Génère un tableau comparatif visuel."""
    archs = list(results.keys())
    configs = list(results[archs[0]].keys()) if archs else []
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    data = []
    for config in configs:
        row = [config]
        for arch in archs:
            acc = results[arch].get(config, {}).get('accuracy', 0)
            row.append(f'{acc:.4f}')
        data.append(row)
    
    table = ax.table(cellText=data, rowLabels=configs, 
                     colLabels=['Config'] + [a.upper() for a in archs],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'comparison_table.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path