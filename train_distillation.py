"""
train_distillation.py — Partie 3 : Knowledge Distillation.

Teacher : GIN centralisé + TACO-Net centralisé (chargés depuis reports/)
Student : StudentGIN + StudentTACO (≥4× moins de paramètres)

Paramètres fixés : alpha=0.5, temperature=4

Usage :
    python train_distillation.py [--epochs N] [--quick]
    (Doit être lancé APRÈS train_centralized.py)
"""

import os
import sys
import time
import json
import argparse
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from dataset      import get_dataloaders, NUM_CLASSES, SELECTED_CLASSES
from models       import get_model, get_student
from distillation import DistillationTrainer
from utils        import (count_parameters, evaluate_model,
                           plot_distillation_comparison,
                           plot_accuracy_bar_chart, plot_params_bar_chart,
                           plot_training_time_bar,
                           print_summary_table, save_results_json, REPORTS_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, 'labeled_dataset.csv')
DEVICE   = torch.device('cpu')

# Hyperparamètres de distillation (fixés)
ALPHA       = 0.5
TEMPERATURE = 4.0


def parse_args():
    p = argparse.ArgumentParser(description='Knowledge Distillation — GIN & TACO-Net')
    p.add_argument('--epochs',  type=int,   default=25,
                   help='Epochs distillation (défaut: 25)')
    p.add_argument('--batch',   type=int,   default=8)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--max_cls', type=int,   default=500)
    p.add_argument('--patience',type=int,   default=5)
    p.add_argument('--quick',   action='store_true',
                   help='Mode rapide : 3 epochs')
    return p.parse_args()


def load_teacher(arch: str, num_classes: int) -> torch.nn.Module:
    """
    Charge le teacher pré-entraîné depuis reports/.
    Si le checkpoint n'existe pas, entraîne un teacher rapide.
    """
    ckpt_path = os.path.join(REPORTS_DIR, f'best_{arch}.pt')
    model     = get_model(arch, num_classes=num_classes)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"  [Teacher {arch.upper()}] Chargé depuis {ckpt_path}")
    else:
        print(f"  [Teacher {arch.upper()}] Checkpoint non trouvé — utilisez d'abord "
              f"train_centralized.py pour entraîner le teacher.")
        print(f"  Attention : le teacher sera non-entraîné (poids aléatoires).")

    return model.to(DEVICE).eval()


def run_distillation(arch: str, train_loader, val_loader, test_loader,
                     n_epochs: int, lr: float, patience: int) -> tuple:
    """Lance la distillation pour une architecture donnée."""
    arch_label = 'GIN' if arch == 'gin' else 'TACO-Net'
    print(f"\n{'─'*60}")
    print(f"  Distillation — Teacher: {arch_label} → Student allégé")

    # Chargement du teacher
    teacher = load_teacher(arch, NUM_CLASSES)

    # Création du student
    student    = get_student(arch, num_classes=NUM_CLASSES).to(DEVICE)
    n_teacher  = count_parameters(teacher)
    n_student  = count_parameters(student)
    ratio      = n_teacher / max(n_student, 1)

    print(f"  Teacher : {n_teacher:,} paramètres")
    print(f"  Student : {n_student:,} paramètres")
    print(f"  Ratio   : {ratio:.1f}× (objectif ≥ 4×) {'✓' if ratio >= 4 else '⚠'}")
    print(f"  α={ALPHA} | τ={TEMPERATURE}")

    # Accuracy du teacher sur val
    teacher_metrics = evaluate_model(teacher, val_loader, DEVICE, SELECTED_CLASSES)
    teacher_acc_val = teacher_metrics['accuracy']
    print(f"  Teacher Val Acc = {teacher_acc_val:.4f}")

    # Entraînement par distillation
    trainer = DistillationTrainer(
        teacher      = teacher,
        student      = student,
        train_loader = train_loader,
        val_loader   = val_loader,
        temperature  = TEMPERATURE,
        alpha        = ALPHA,
        lr           = lr,
        n_epochs     = n_epochs,
        patience     = patience,
        device       = DEVICE,
    )
    history = trainer.run(teacher_acc=teacher_acc_val)

    # Évaluation finale sur test
    teacher_test = evaluate_model(teacher, test_loader, DEVICE, SELECTED_CLASSES)
    student_test = evaluate_model(student, test_loader, DEVICE, SELECTED_CLASSES)

    print(f"\n  Teacher Test Acc = {teacher_test['accuracy']:.4f}")
    print(f"  Student Test Acc = {student_test['accuracy']:.4f}")

    # Sauvegarde
    os.makedirs(REPORTS_DIR, exist_ok=True)
    torch.save(student.state_dict(),
               os.path.join(REPORTS_DIR, f'student_{arch}.pt'))
    with open(os.path.join(REPORTS_DIR, f'distill_{arch}_history.json'), 'w') as f:
        json.dump(history, f)

    return history, teacher_test, student_test, n_teacher, n_student


def main():
    args = parse_args()

    if args.quick:
        args.epochs  = 3
        args.max_cls = 50
        args.patience = 2
        print("\n  [QUICK] Mode rapide : 3 epochs distillation\n")

    print(f"\n{'#'*60}")
    print(f"  PARTIE 3 — Knowledge Distillation")
    print(f"  α = {ALPHA} | τ = {TEMPERATURE}")
    print(f"  Epochs max : {args.epochs} | Patience : {args.patience}")
    print(f"  Device : {DEVICE}")
    print(f"{'#'*60}\n")

    # ── Chargement des données ────────────────────────────────────────────────
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        CSV_PATH, max_per_class=args.max_cls, batch_size=args.batch)

    all_results = {}
    all_params  = {}
    all_times   = {}
    gin_hist    = None
    taco_hist   = None

    # ── Distillation GIN ──────────────────────────────────────────────────────
    gin_hist, teacher_gin, student_gin, n_t_gin, n_s_gin = run_distillation(
        'gin', train_loader, val_loader, test_loader,
        n_epochs=args.epochs, lr=args.lr, patience=args.patience)

    # ── Distillation TACO-Net ─────────────────────────────────────────────────
    taco_hist, teacher_taco, student_taco, n_t_taco, n_s_taco = run_distillation(
        'taconet', train_loader, val_loader, test_loader,
        n_epochs=args.epochs, lr=args.lr, patience=args.patience)

    # ── Consolidation des résultats ───────────────────────────────────────────
    # Résultats teachers
    all_results['GIN Teacher']       = {**teacher_gin,
                                        'n_params': n_t_gin,
                                        'train_time': 0}
    all_results['GIN Student']       = {**student_gin,
                                        'n_params': n_s_gin,
                                        'train_time': gin_hist.get('total_time', 0)}
    all_results['TACO-Net Teacher']  = {**teacher_taco,
                                        'n_params': n_t_taco,
                                        'train_time': 0}
    all_results['TACO-Net Student']  = {**student_taco,
                                        'n_params': n_s_taco,
                                        'train_time': taco_hist.get('total_time', 0)}

    all_params = {
        'GIN Teacher':      n_t_gin,
        'GIN Student':      n_s_gin,
        'TACO Teacher':     n_t_taco,
        'TACO Student':     n_s_taco,
    }
    all_times = {
        'GIN Student': gin_hist.get('total_time', 0),
        'TACO Student': taco_hist.get('total_time', 0),
    }

    # ── Génération des graphiques ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Génération des graphiques distillation...")

    # Graphique 6 : Distillation comparison
    plot_distillation_comparison(gin_hist, taco_hist)

    # Graphique 7 : Accuracy comparison (teachers + students)
    acc_dict = {
        'GIN Teacher':     teacher_gin['accuracy'],
        'GIN Student':     student_gin['accuracy'],
        'TACO Teacher':    teacher_taco['accuracy'],
        'TACO Student':    student_taco['accuracy'],
    }
    plot_accuracy_bar_chart(acc_dict)

    # Graphique 8 : Parameter count
    plot_params_bar_chart(all_params)

    # Graphique 9 : Training time
    plot_training_time_bar(all_times)

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    print_summary_table(all_results)
    save_results_json(all_results, 'distillation_results.json')

    print(f"\n{'#'*60}")
    print(f"  ✅ Partie 3 terminée. Graphiques → plots/")
    print(f"  ✅ Résultats      → reports/")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
