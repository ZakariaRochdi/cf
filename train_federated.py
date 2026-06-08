"""
train_federated.py — Partie 2 : Federated Learning.

Implémente :
  - HFL (Horizontal Federated Learning) avec FedAvg manuel
  - VFL (Vertical Federated Learning) → Entité A (XYZ) + Entité B (RGB)

Usage :
    python train_federated.py [--rounds N] [--local_ep N] [--quick]
"""

import os
import sys
import time
import json
import argparse
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from dataset    import (get_dataloaders, build_hfl_clients,
                         NUM_CLASSES, SELECTED_CLASSES, CLIENT_NAMES)
from models     import get_model
from federated  import HorizontalFL, VerticalFL
from utils      import (count_parameters, evaluate_model,
                         plot_fedavg_convergence, plot_accuracy_bar_chart,
                         plot_training_time_bar,
                         print_summary_table, save_results_json, REPORTS_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, 'labeled_dataset.csv')
DEVICE   = torch.device('cpu')


def parse_args():
    p = argparse.ArgumentParser(description='Federated Learning — HFL & VFL')
    p.add_argument('--rounds',    type=int,   default=5)
    p.add_argument('--local_ep',  type=int,   default=2)
    p.add_argument('--vfl_ep',    type=int,   default=15)
    p.add_argument('--batch',     type=int,   default=8)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--max_cls',   type=int,   default=500)
    p.add_argument('--skip_vfl',  action='store_true')
    p.add_argument('--quick',     action='store_true')
    return p.parse_args()


def run_hfl(arch: str, train_df, val_loader, test_loader,
            n_rounds: int, local_epochs: int, lr: float,
            batch_size: int, iid: bool = False) -> tuple:
    """Lance HFL pour une architecture donnée."""
    arch_label = 'GIN' if arch == 'gin' else 'TACO-Net'
    mode_label = 'IID' if iid else 'non-IID'
    print(f"\n{'─'*60}")
    print(f"  HFL — {arch_label} ({n_rounds} rounds, {local_epochs} epochs locales, {mode_label})")

    client_loaders = build_hfl_clients(train_df, batch_size=batch_size, iid=iid)

    global_model = get_model(arch, num_classes=NUM_CLASSES).to(DEVICE)
    print(f"  Paramètres modèle : {count_parameters(global_model):,}")

    fl = HorizontalFL(
        global_model   = global_model,
        client_loaders = client_loaders,
        val_loader     = val_loader,
        n_rounds       = n_rounds,
        local_epochs   = local_epochs,
        lr             = lr,
        device         = DEVICE,
        client_names   = CLIENT_NAMES,
    )
    history = fl.run()

    per_client = fl.get_per_client_accuracy()
    print(f"\n  Accuracy par client ({arch_label}) :")
    for name, acc in per_client.items():
        print(f"    {name:<15}: {acc:.4f}")

    metrics = evaluate_model(fl.global_model, test_loader, DEVICE, SELECTED_CLASSES)
    metrics['train_time'] = history.get('total_time', 0)
    metrics['n_params']   = count_parameters(fl.global_model)
    metrics['per_client'] = per_client

    os.makedirs(REPORTS_DIR, exist_ok=True)
    torch.save(fl.global_model.state_dict(),
               os.path.join(REPORTS_DIR, f'hfl_{arch}_{mode_label}_global.pt'))
    with open(os.path.join(REPORTS_DIR, f'hfl_{arch}_{mode_label}_history.json'), 'w') as f:
        json.dump(history, f)

    return fl.global_model, history, metrics


def run_vfl(train_loader, val_loader, test_loader,
            n_epochs: int, lr: float) -> tuple:
    """Lance VFL (XYZ + RGB)."""
    print(f"\n{'─'*60}")
    print(f"  VFL — Entité A (XYZ) + Entité B (RGB)")

    vfl = VerticalFL(
        train_loader = train_loader,
        val_loader   = val_loader,
        num_classes  = NUM_CLASSES,
        embed_dim    = 64,
        lr           = lr,
        n_epochs     = n_epochs,
        device       = DEVICE,
    )
    history = vfl.run()

    @torch.no_grad()
    def vfl_predict(loader):
        vfl.enc_A.eval(); vfl.enc_B.eval(); vfl.server.eval()
        correct, total = 0, 0
        for pts, labels in loader:
            pts, labels = pts.to(DEVICE), labels.to(DEVICE)
            logits = vfl.server(vfl.enc_A(pts[:, :3, :]), vfl.enc_B(pts[:, 3:, :]))
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        return correct / total

    test_acc = vfl_predict(test_loader)
    n_params = (count_parameters(vfl.enc_A) +
                count_parameters(vfl.enc_B) +
                count_parameters(vfl.server))

    metrics = {
        'accuracy':   test_acc,
        'train_time': history.get('total_time', 0),
        'n_params':   n_params,
    }
    print(f"\n  ✓ VFL Test Accuracy = {test_acc:.4f}")
    print(f"  ✓ Total params (A+B+Server) = {n_params:,}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, 'vfl_history.json'), 'w') as f:
        json.dump(history, f)

    return history, metrics


def main():
    args = parse_args()

    if args.quick:
        args.rounds   = 2
        args.local_ep = 1
        args.vfl_ep   = 3
        args.max_cls  = 50
        print("\n  [QUICK] Mode rapide : 2 rounds, 1 epoch locale\n")

    print(f"\n{'#'*60}")
    print(f"  PARTIE 2 — Federated Learning")
    print(f"  HFL : {args.rounds} rounds | {args.local_ep} epochs locales")
    if not args.skip_vfl:
        print(f"  VFL : {args.vfl_ep} epochs")
    print(f"  Device : {DEVICE}")
    print(f"{'#'*60}\n")

    train_loader, val_loader, test_loader, train_df, _, _ = get_dataloaders(
        CSV_PATH, max_per_class=args.max_cls, batch_size=args.batch)

    all_results = {}
    all_times   = {}
    gin_hist    = None
    taco_hist   = None

    for arch in ['gin', 'taconet']:
        arch_label = 'GIN' if arch == 'gin' else 'TACO-Net'

        # IID
        _, hist_iid, metrics_iid = run_hfl(
            arch, train_df, val_loader, test_loader,
            n_rounds=args.rounds, local_epochs=args.local_ep,
            lr=args.lr, batch_size=args.batch, iid=True)

        label_iid = f'{arch_label} HFL-IID'
        all_results[label_iid] = metrics_iid
        all_times[label_iid]   = metrics_iid.get('train_time', 0)

        # non-IID
        _, hist_noniid, metrics_noniid = run_hfl(
            arch, train_df, val_loader, test_loader,
            n_rounds=args.rounds, local_epochs=args.local_ep,
            lr=args.lr, batch_size=args.batch, iid=False)

        label_noniid = f'{arch_label} HFL-nonIID'
        all_results[label_noniid] = metrics_noniid
        all_times[label_noniid]   = metrics_noniid.get('train_time', 0)

        if arch == 'gin':
            gin_hist = hist_iid
        else:
            taco_hist = hist_iid

        print(f"\n  [{arch_label}] IID    Acc = {metrics_iid['accuracy']:.4f}")
        print(f"  [{arch_label}] nonIID Acc = {metrics_noniid['accuracy']:.4f}")

    if not args.skip_vfl:
        vfl_hist, vfl_metrics = run_vfl(
            train_loader, val_loader, test_loader,
            n_epochs=args.vfl_ep, lr=args.lr)
        all_results['VFL (XYZ+RGB)'] = vfl_metrics
        all_times['VFL']             = vfl_metrics.get('train_time', 0)
        print(f"\n  [VFL] Test Acc = {vfl_metrics['accuracy']:.4f}")

    print(f"\n{'─'*60}")
    print("  Génération des graphiques FedAvg...")
    plot_fedavg_convergence(gin_hist, taco_hist)
    acc_dict = {k: v.get('accuracy', 0) for k, v in all_results.items()}
    plot_accuracy_bar_chart(acc_dict)
    plot_training_time_bar(all_times)

    print_summary_table(all_results)
    save_results_json(all_results, 'federated_results.json')

    print(f"\n{'#'*60}")
    print(f"  ✅ Partie 2 terminée. Résultats → reports/")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()