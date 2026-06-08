"""
main.py — Point d'entrée principal du projet.
Lance séquentiellement toutes les expériences C1–C4 pour les deux architectures
(GIN et TACO-Net), puis génère les courbes et le tableau comparatif.

Usage :
    python main.py [--quick] [--ply_dir PATH] [--epochs N] [--rounds N]
"""

import os, sys, argparse, json, copy
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(PROJECT_DIR, 'labeled_dataset.csv')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
sys.path.insert(0, PROJECT_DIR)

from dataset            import get_dataloaders, NUM_CLASSES
from models             import get_model
from train_centralized  import train_model
from train_federated    import run_hfl
from train_distillation import run_distillation
from federated.vfl      import VerticalFL
from utils              import (evaluate_model, plot_convergence,
                                plot_comparison_table, plot_per_class,
                                plot_confusion_matrix, print_summary_table,
                                save_results_json, count_parameters)


def parse_args():
    p = argparse.ArgumentParser(description='Classification 3D — Graph CNN')
    p.add_argument('--quick',     action='store_true')
    p.add_argument('--ply_dir',   type=str,   default=None)
    p.add_argument('--epochs',    type=int,   default=40)
    p.add_argument('--rounds',    type=int,   default=20)
    p.add_argument('--local_ep',  type=int,   default=3)
    p.add_argument('--max_cls',   type=int,   default=200)
    p.add_argument('--batch',     type=int,   default=32)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--patience',  type=int,   default=10)
    p.add_argument('--archs',     nargs='+',  default=['gin', 'taconet'])
    p.add_argument('--skip_vfl',  action='store_true')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'#'*60}")
    print(f"  Classification de Nuages de Points 3D")
    print(f"  Device  : {device}")
    print(f"  Archs   : {args.archs}")
    print(f"  CSV     : {CSV_PATH}")
    print(f"  PLY dir : {args.ply_dir or '(génération synthétique)'}")
    print(f"{'#'*60}\n")

    if args.quick:
        args.epochs   = 2
        args.rounds   = 2
        args.local_ep = 1
        args.max_cls  = 30
        args.patience = 2
        print("  [RAPIDE] Mode rapide activé (2 epochs, 2 rounds, 30 obj/classe)\n")

    # ── Chargement des données ─────────────────────────────────────────────────
    print("  Chargement des données...")
    train_loader, val_loader, test_loader, train_df, val_df, test_df = \
        get_dataloaders(CSV_PATH, ply_dir=args.ply_dir,
                        max_per_class=args.max_cls,
                        batch_size=args.batch)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results   = {}
    histories = {}

    # ==========================================================================
    # BOUCLE SUR LES ARCHITECTURES
    # ==========================================================================
    for arch in args.archs:
        print(f"\n{'-'*60}")
        print(f"  ARCHITECTURE : {arch.upper()}")
        print(f"{'-'*60}")
        results[arch]   = {}
        histories[arch] = {}

        # ── C1 : Baseline centralisée ──────────────────────────────────────────
        print("\n> C1 — Baseline Centralisée")
        model_c1, hist_c1, _ = train_model(
            arch, train_loader, val_loader,
            n_epochs=args.epochs, lr=args.lr, patience=args.patience)

        histories[arch]['C1 Baseline'] = hist_c1
        results[arch]['C1 Baseline']   = evaluate_model(model_c1, test_loader, device)
        print(f"  → C1 Test Acc : {results[arch]['C1 Baseline']['accuracy']:.4f}")

        # ── C2 : Fédération horizontale IID ────────────────────────────────────
        print("\n> C2 — Fédération Horizontale (IID)")
        model_c2, hist_c2, _ = run_hfl(
            arch, train_df, val_loader, test_loader,
            n_rounds=args.rounds,
            local_epochs=args.local_ep,
            lr=args.lr,
            batch_size=args.batch,
            iid=True)

        histories[arch]['C2 Fed-IID'] = hist_c2
        results[arch]['C2 Fed-IID']   = evaluate_model(model_c2, test_loader, device)
        print(f"  → C2 Test Acc : {results[arch]['C2 Fed-IID']['accuracy']:.4f}")

        # ── C3 : Fédération horizontale non-IID ────────────────────────────────
        print("\n> C3 — Fédération Horizontale (non-IID)")
        model_c3, hist_c3, _ = run_hfl(
            arch, train_df, val_loader, test_loader,
            n_rounds=args.rounds,
            local_epochs=args.local_ep,
            lr=args.lr,
            batch_size=args.batch,
            iid=False)

        histories[arch]['C3 Fed-nonIID'] = hist_c3
        results[arch]['C3 Fed-nonIID']   = evaluate_model(model_c3, test_loader, device)
        print(f"  → C3 Test Acc : {results[arch]['C3 Fed-nonIID']['accuracy']:.4f}")

        # ── C4 : Distillation de connaissances ─────────────────────────────────
        print("\n> C4 — Distillation de Connaissances")
        hist_c4, _, student_metrics, _, _ = run_distillation(
            arch, train_loader, val_loader, test_loader,
            n_epochs=args.epochs,
            lr=args.lr,
            patience=args.patience)

        histories[arch]['C4 Distill'] = hist_c4
        results[arch]['C4 Distill']   = student_metrics
        print(f"  → C4 Test Acc : {results[arch]['C4 Distill']['accuracy']:.4f}")

        # ── Courbes par architecture ────────────────────────────────────────────
        plot_convergence(histories[arch], arch, save_dir=RESULTS_DIR)
        plot_per_class(results, arch, save_dir=RESULTS_DIR)
        plot_confusion_matrix(model_c1, test_loader, device, arch, 'C1',
                              save_dir=RESULTS_DIR)

    # ── Fédération Verticale (optionnelle) ─────────────────────────────────────
    if not args.skip_vfl:
        print(f"\n{'-'*60}")
        print("  FÉDÉRATION VERTICALE (XYZ / RGB)")
        print(f"{'-'*60}")
        vfl = VerticalFL(train_loader, val_loader,
                         num_classes=NUM_CLASSES,
                         n_epochs=args.epochs,
                         device=device)
        vfl_history = vfl.run()
        with open(os.path.join(RESULTS_DIR, 'vfl_history.json'), 'w') as f:
            json.dump(vfl_history, f)

    # ── Tableau comparatif global ──────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("  COMPARAISON GLOBALE")
    print(f"{'-'*60}")
    print_summary_table(results)
    plot_comparison_table(results, save_dir=RESULTS_DIR)
    save_results_json(results, save_dir=RESULTS_DIR)

    print(f"\n{'#'*60}")
    print(f"  ✅ Projet terminé. Résultats dans : {RESULTS_DIR}")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()