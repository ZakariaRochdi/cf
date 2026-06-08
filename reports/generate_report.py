"""
reports/generate_report.py — Génération automatique du rapport PDF.

Génère un rapport PDF académique complet contenant :
  - Page de titre
  - Description des architectures GIN et TACO-Net
  - Méthodologie (dataset, FL, distillation)
  - Résultats des expériences (tableaux + graphiques intégrés)
  - Conclusion

Usage :
    python reports/generate_report.py
    (Doit être lancé APRÈS les 3 scripts d'entraînement)
"""

import os
import sys
import json
import glob
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np

REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
PLOTS_DIR   = os.path.join(PROJECT_DIR, 'plots')
PDF_PATH    = os.path.join(REPORTS_DIR, 'rapport_final.pdf')


# ─── Utilitaires de mise en page ──────────────────────────────────────────────
def new_page(pdf: PdfPages, figsize=(11.69, 8.27)) -> tuple:
    """Crée une nouvelle page (format A4 paysage)."""
    fig = plt.figure(figsize=figsize)
    return fig, pdf


def add_header(fig, title: str, page_num: int = None):
    """Ajoute un en-tête à la page."""
    fig.text(0.5, 0.97, title, ha='center', va='top',
             fontsize=15, fontweight='bold', color='#1e3a5f')
    fig.text(0.05, 0.97,
             'Classification de Nuages de Points 3D', ha='left', va='top',
             fontsize=9, color='#6b7280', style='italic')
    if page_num:
        fig.text(0.95, 0.97, f'Page {page_num}', ha='right', va='top',
                 fontsize=9, color='#6b7280')
    # Ligne de séparation
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.955, 0.955],
                               color='#1e3a5f', lw=1.5,
                               transform=fig.transFigure))


def load_results(filename: str) -> dict:
    """Charge les résultats JSON depuis reports/."""
    path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_history(filename: str) -> dict:
    """Charge un historique JSON depuis reports/."""
    path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def get_plot(filename: str):
    """Retourne l'image matplotlib depuis plots/ si elle existe."""
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        return plt.imread(path)
    return None


# ─── Pages du rapport ─────────────────────────────────────────────────────────

def page_title(pdf: PdfPages):
    """Page 1 : Page de titre."""
    fig = plt.figure(figsize=(11.69, 8.27))

    # Fond coloré
    bg = FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0',
                         facecolor='#1e3a5f', edgecolor='none',
                         transform=fig.transFigure)
    fig.add_artist(bg)

    fig.text(0.5, 0.72,
             'Classification de Nuages de Points 3D',
             ha='center', va='center', fontsize=24, fontweight='bold',
             color='white')
    fig.text(0.5, 0.62,
             'Graph CNN · Federated Learning · Knowledge Distillation',
             ha='center', va='center', fontsize=16, color='#93c5fd')
    fig.text(0.5, 0.50,
             '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
             ha='center', va='center', fontsize=12, color='#93c5fd')
    fig.text(0.5, 0.40,
             'Architectures : GIN (Graph Isomorphism Network)  ·  TACO-Net',
             ha='center', va='center', fontsize=13, color='white')
    fig.text(0.5, 0.33,
             'Dataset : ShapeNet — 8 classes · 512 points · CPU-only',
             ha='center', va='center', fontsize=12, color='#d1d5db')
    fig.text(0.5, 0.22,
             f'Rapport généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}',
             ha='center', va='center', fontsize=10, color='#9ca3af', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_intro(pdf: PdfPages):
    """Page 2 : Introduction et Description des architectures."""
    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(fig, 'Introduction & Architectures', 2)

    ax = fig.add_axes([0.05, 0.05, 0.9, 0.87])
    ax.axis('off')

    text = """
CONTEXTE DU PROJET
══════════════════
Ce mini-projet universitaire réalise une expérimentation complète de classification de nuages de points 3D
avec trois composantes principales : (1) baseline centralisée, (2) apprentissage fédéré, (3) distillation
de connaissances. Toutes les expériences sont optimisées pour un entraînement sur CPU uniquement.

• Dataset    : Sous-ensemble ShapeNet — 8 classes, max 500 objets/classe, 512 points/objet
• Split      : Train 70% / Val 15% / Test 15% (seed=42, reproductible)
• Augmentations : rotation Z, jitter faible (σ=0.01), scaling [0.9, 1.1]

ARCHITECTURE 1 — GIN (Graph Isomorphism Network)
══════════════════════════════════════════════════
GIN [Xu et al., ICLR 2019] est l'une des GNN les plus expressives. Pour un nœud v, la mise
à jour est : h_v^(l+1) = MLP((1+ε)·h_v^(l) + Σ_{u∈N(v)} h_u^(l))

Implémentation CPU-légère :
  • 3 couches GINConv avec MLP(in→64→64) | K-NN k=10 (sur XYZ)
  • Agrégation globale : Mean Pool + Max Pool → concat(128)
  • FC classifier : 128 → 64 → 8 classes | Dropout=0.5
  • ~200K paramètres

ARCHITECTURE 2 — TACO-Net (Token-based Attention with Compact Operations)
═══════════════════════════════════════════════════════════════════════════
TACO-Net utilise une attention locale sur les voisins k-NN, analogue à un Transformer
limité à un voisinage local pour réduire la complexité O(N²) → O(N·k).

Implémentation CPU-légère :
  • Projection initiale : 6→32 via Conv1D
  • 2 blocs d'attention locale (k=10) avec projections Q,K,V (32→64)
  • Agrégation : Max Pool + Mean Pool → concat(128) → FC(128→64→8)
  • ~150K paramètres
"""
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            color='#111827', linespacing=1.6)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_methodology(pdf: PdfPages):
    """Page 3 : Méthodologie."""
    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(fig, 'Méthodologie', 3)

    ax = fig.add_axes([0.05, 0.05, 0.9, 0.87])
    ax.axis('off')

    text = """
PARTIE 1 — BASELINE CENTRALISÉE
══════════════════════════════════
Entraînement indépendant de GIN et TACO-Net sur l'ensemble des données centralisées.

  • Optimiseur : Adam (lr=0.001, weight_decay=1e-4)
  • Loss : CrossEntropyLoss avec label_smoothing=0.05
  • Scheduler : CosineAnnealingLR
  • Epochs max : 30 | Early Stopping : patience=5 (sur Val Accuracy)
  • Métriques : Accuracy, Precision, Recall, F1-score (macro), Temps, Params

PARTIE 2 — FEDERATED LEARNING
══════════════════════════════
[HFL] Federated Horizontal Learning — FedAvg manuel :
  • 4 clients avec partition sémantique non-IID :
    - Client 0 (Transport)   : airplane, car
    - Client 1 (Mobilier)    : chair, table
    - Client 2 (Cylindrique) : lamp, bottle
    - Client 3 (Lounge)      : bed, sofa
  • FedAvg : w^(t+1) = Σ_k (n_k/n)·w_k^(t)  [moyenne pondérée par taille dataset]
  • Implémentation : state_dict + copy.deepcopy uniquement (pas de Flower/FedML)
  • 5 rounds | 2 epochs locales par round

[VFL] Federated Vertical Learning — partition par attributs :
  • Entité A (XYZ) : XYZEncoder → embedding e_A ∈ R^64
  • Entité B (RGB) : RGBEncoder → embedding e_B ∈ R^64
  • Serveur : concat[e_A||e_B] → FC → logits (15 epochs)

PARTIE 3 — KNOWLEDGE DISTILLATION
════════════════════════════════════
Formule combinée [Hinton et al., 2015] :
  L = (1-α)·CE(ŷ_student, y) + α·τ²·KL(σ(z_s/τ) || σ(z_t/τ))

Avec : α = 0.5 (pondération KL/CE) | τ = 4 (température)

  • Teacher : GIN + TACO-Net centralisés (pré-entraînés)
  • Student : StudentGIN (2 couches, hidden=16) + StudentTACO (1 bloc, embed=16)
  • Ratio paramètres teacher/student : ≥ 4× (objectif imposé)
  • Early stopping : patience=5 sur Val Accuracy du student
"""
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            color='#111827', linespacing=1.6)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_results_centralized(pdf: PdfPages):
    """Page 4 : Résultats de la baseline centralisée + graphiques."""
    results = load_results('centralized_results.json')

    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(fig, 'Résultats — Partie 1 : Baseline Centralisée', 4)

    if not results:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        ax.text(0.5, 0.5, 'Résultats non disponibles.\nLancez train_centralized.py',
                ha='center', va='center', fontsize=14, color='gray')
        pdf.savefig(fig); plt.close(fig); return

    # Tableau des résultats
    ax_table = fig.add_axes([0.05, 0.60, 0.90, 0.30])
    ax_table.axis('off')

    cols  = ['Modèle', 'Accuracy', 'Precision', 'Recall', 'F1', 'Paramètres', 'Temps']
    rows  = []
    for name, m in results.items():
        rows.append([
            name,
            f"{m.get('accuracy',0)*100:.2f}%",
            f"{m.get('precision',0)*100:.2f}%",
            f"{m.get('recall',0)*100:.2f}%",
            f"{m.get('f1',0)*100:.2f}%",
            f"{m.get('n_params',0):,}",
            f"{m.get('train_time',0)/60:.1f} min",
        ])

    tbl = ax_table.table(cellText=rows, colLabels=cols,
                         cellLoc='center', loc='center',
                         colColours=['#1e3a5f']*len(cols))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f0f4ff')

    # Graphiques Loss/Accuracy
    gin_img  = get_plot('gin_loss_curve.png')
    taco_img = get_plot('taconet_loss_curve.png')

    if gin_img is not None:
        ax1 = fig.add_axes([0.05, 0.05, 0.43, 0.50])
        ax1.imshow(gin_img); ax1.axis('off')
        ax1.set_title('GIN — Loss & Accuracy', fontsize=10)

    if taco_img is not None:
        ax2 = fig.add_axes([0.52, 0.05, 0.43, 0.50])
        ax2.imshow(taco_img); ax2.axis('off')
        ax2.set_title('TACO-Net — Loss & Accuracy', fontsize=10)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_results_federated(pdf: PdfPages):
    """Page 5 : Résultats du Federated Learning."""
    results = load_results('federated_results.json')

    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(fig, 'Résultats — Partie 2 : Federated Learning', 5)

    if not results:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        ax.text(0.5, 0.5, 'Résultats non disponibles.\nLancez train_federated.py',
                ha='center', va='center', fontsize=14, color='gray')
        pdf.savefig(fig); plt.close(fig); return

    # Tableau
    ax_table = fig.add_axes([0.05, 0.62, 0.90, 0.28])
    ax_table.axis('off')

    cols = ['Configuration', 'Accuracy', 'F1', 'Paramètres', 'Temps']
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            f"{m.get('accuracy',0)*100:.2f}%",
            f"{m.get('f1',0)*100:.2f}%",
            f"{m.get('n_params',0):,}",
            f"{m.get('train_time',0)/60:.1f} min",
        ])

    tbl = ax_table.table(cellText=rows, colLabels=cols,
                         cellLoc='center', loc='center',
                         colColours=['#1e3a5f']*len(cols))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f0f4ff')

    # Graphique convergence FedAvg
    fed_img = get_plot('fedavg_convergence_curve.png')
    if fed_img is not None:
        ax_fed = fig.add_axes([0.15, 0.05, 0.70, 0.50])
        ax_fed.imshow(fed_img); ax_fed.axis('off')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_results_distillation(pdf: PdfPages):
    """Page 6 : Résultats de la Knowledge Distillation."""
    results = load_results('distillation_results.json')

    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(fig, 'Résultats — Partie 3 : Knowledge Distillation', 6)

    if not results:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        ax.text(0.5, 0.5, 'Résultats non disponibles.\nLancez train_distillation.py',
                ha='center', va='center', fontsize=14, color='gray')
        pdf.savefig(fig); plt.close(fig); return

    # Tableau comparatif Teacher vs Student
    ax_table = fig.add_axes([0.05, 0.62, 0.90, 0.28])
    ax_table.axis('off')

    cols = ['Modèle', 'Rôle', 'Accuracy', 'F1', 'Paramètres', 'Ratio']
    rows = []

    items = list(results.items())
    params_by_arch = {}
    for name, m in items:
        params_by_arch[name] = m.get('n_params', 1)

    for name, m in items:
        role = 'Teacher' if 'Teacher' in name else 'Student (distillé)'
        # Ratio teacher/student
        if 'Student' in name:
            arch = name.split()[0]
            t_key = f"{arch} Teacher"
            t_params = params_by_arch.get(t_key, 1)
            s_params = m.get('n_params', 1)
            ratio = f"{t_params/max(s_params,1):.1f}×"
        else:
            ratio = '—'

        rows.append([
            name, role,
            f"{m.get('accuracy',0)*100:.2f}%",
            f"{m.get('f1',0)*100:.2f}%",
            f"{m.get('n_params',0):,}",
            ratio,
        ])

    tbl = ax_table.table(cellText=rows, colLabels=cols,
                         cellLoc='center', loc='center',
                         colColours=['#1e3a5f']*len(cols))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f0f4ff')

    # Graphique distillation
    dist_img = get_plot('distillation_comparison_curve.png')
    if dist_img is not None:
        ax_dist = fig.add_axes([0.10, 0.05, 0.80, 0.50])
        ax_dist.imshow(dist_img); ax_dist.axis('off')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_comparison(pdf: PdfPages):
    """Page 7 : Tableau comparatif global + graphiques bar charts."""
    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(fig, 'Comparaison Globale — Toutes Configurations', 7)

    imgs = {
        'accuracy': get_plot('accuracy_comparison_bar.png'),
        'params':   get_plot('parameter_count_bar.png'),
        'time':     get_plot('training_time_bar.png'),
    }

    positions = [
        ([0.03, 0.05, 0.30, 0.85], 'accuracy', 'Accuracy (%)'),
        ([0.36, 0.05, 0.30, 0.85], 'params',   'Paramètres (K)'),
        ([0.68, 0.05, 0.30, 0.85], 'time',     'Temps (min)'),
    ]

    for (pos, key, title) in positions:
        ax = fig.add_axes(pos)
        if imgs[key] is not None:
            ax.imshow(imgs[key]); ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'{title}\n(données manquantes)',
                    ha='center', va='center', fontsize=10, color='gray')
            ax.axis('off')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_confusion_matrices(pdf: PdfPages):
    """Page 8 : Matrices de confusion."""
    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(fig, 'Matrices de Confusion', 8)

    # Chercher les images de confusion matrices
    conf_imgs = []
    for arch in ['gin_baseline', 'taconet_baseline']:
        img = get_plot(f'confusion_{arch}.png')
        if img is None:
            # Essayer d'autres noms possibles
            for fname in os.listdir(PLOTS_DIR) if os.path.exists(PLOTS_DIR) else []:
                if 'confusion' in fname.lower() and arch.split('_')[0] in fname.lower():
                    img = get_plot(fname)
                    break
        if img is not None:
            conf_imgs.append((img, arch.replace('_', ' ').title()))

    if not conf_imgs:
        # Chercher toutes les confusion matrices disponibles
        if os.path.exists(PLOTS_DIR):
            for fname in sorted(os.listdir(PLOTS_DIR)):
                if 'confusion' in fname.lower():
                    img = get_plot(fname)
                    if img is not None:
                        conf_imgs.append((img, fname.replace('.png', '').replace('_', ' ').title()))

    if conf_imgs:
        n = min(len(conf_imgs), 2)
        width = 0.85 / n
        for i, (img, title) in enumerate(conf_imgs[:n]):
            ax = fig.add_axes([0.05 + i * (width + 0.05), 0.05, width, 0.85])
            ax.imshow(img); ax.axis('off')
            ax.set_title(title, fontsize=11)
    else:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        ax.text(0.5, 0.5, 'Matrices de confusion\nnon disponibles',
                ha='center', va='center', fontsize=14, color='gray')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_conclusion(pdf: PdfPages):
    """Page 9 : Conclusion."""
    # Charger résultats pour synthèse
    cent = load_results('centralized_results.json')
    fed  = load_results('federated_results.json')
    dist = load_results('distillation_results.json')

    fig = plt.figure(figsize=(11.69, 8.27))
    add_header(fig, 'Conclusion', 9)

    ax = fig.add_axes([0.05, 0.05, 0.90, 0.87])
    ax.axis('off')

    # Synthèse numérique
    gin_acc  = cent.get('GIN Baseline',    {}).get('accuracy', 0) * 100
    taco_acc = cent.get('TACO-Net Baseline',{}).get('accuracy', 0) * 100
    fed_acc  = max([v.get('accuracy', 0) for v in fed.values()], default=0) * 100

    gin_t_acc  = dist.get('GIN Teacher',      {}).get('accuracy', 0) * 100
    gin_s_acc  = dist.get('GIN Student',       {}).get('accuracy', 0) * 100
    taco_t_acc = dist.get('TACO-Net Teacher',  {}).get('accuracy', 0) * 100
    taco_s_acc = dist.get('TACO-Net Student',  {}).get('accuracy', 0) * 100

    text = f"""
SYNTHÈSE DES RÉSULTATS
══════════════════════

1. BASELINE CENTRALISÉE
   ├── GIN      : Accuracy = {gin_acc:.2f}%
   └── TACO-Net : Accuracy = {taco_acc:.2f}%

   Les deux architectures Graph CNN légères (optimisées CPU, ~150–200K paramètres)
   obtiennent des performances compétitives sur les 8 classes ShapeNet sélectionnées.

2. FEDERATED LEARNING (FedAvg manuel)
   ├── HFL GIN      : Meilleure accuracy globale = {fed_acc:.2f}%
   └── VFL (XYZ+RGB) : Séparation stricte attributs géométriques / couleur

   La fédération horizontale avec partition non-IID montre une convergence progressive
   sur 5 rounds. Le VFL démontre la faisabilité de l'apprentissage distribué par
   attributs sans échange de données brutes.

3. KNOWLEDGE DISTILLATION (α=0.5, τ=4)
   ├── GIN      : Teacher={gin_t_acc:.2f}%  →  Student={gin_s_acc:.2f}%
   └── TACO-Net : Teacher={taco_t_acc:.2f}%  →  Student={taco_s_acc:.2f}%

   Les modèles student (≥4× moins de paramètres) conservent une fraction significative
   des performances teacher grâce à la combinaison CE+KL avec température τ=4.

APPORTS ET LIMITES
══════════════════

Apports :
  • Implémentation PyTorch pur (GIN + TACO-Net sans torch-geometric) — reproductible
  • FedAvg entièrement manuel (state_dict uniquement) — pédagogique
  • Pipeline CPU complet : 512 points, batch=8, seed fixe = résultats reproductibles

Limites :
  • Données synthétiques (nuages générés) si .ply non disponibles → biais
  • 8 classes seulement pour respecter les contraintes CPU
  • Performances en dessous de l'état de l'art (objectif pédagogique, non SOTA)

REPRODUCTIBILITÉ
════════════════
  Seed : 42 | Split : 70/15/15 | CPU uniquement | Pas de CUDA requis
  Exécution : python train_centralized.py → train_federated.py → train_distillation.py
"""
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            color='#111827', linespacing=1.6)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─── Génération complète du PDF ───────────────────────────────────────────────
def generate_pdf():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Génération du rapport PDF...")
    print(f"  Output : {PDF_PATH}")
    print(f"{'='*60}\n")

    with PdfPages(PDF_PATH) as pdf:
        # Métadonnées PDF
        d = pdf.infodict()
        d['Title']   = 'Classification de Nuages de Points 3D'
        d['Subject'] = 'Graph CNN, Federated Learning, Knowledge Distillation'
        d['Keywords'] = 'GIN, TACO-Net, FedAvg, Distillation, ShapeNet, CPU'
        d['CreationDate'] = datetime.now()

        page_title(pdf)
        print("  [1/9] Page de titre ✓")

        page_intro(pdf)
        print("  [2/9] Introduction & Architectures ✓")

        page_methodology(pdf)
        print("  [3/9] Méthodologie ✓")

        page_results_centralized(pdf)
        print("  [4/9] Résultats Baseline ✓")

        page_results_federated(pdf)
        print("  [5/9] Résultats Federated Learning ✓")

        page_results_distillation(pdf)
        print("  [6/9] Résultats Distillation ✓")

        page_comparison(pdf)
        print("  [7/9] Comparaison globale ✓")

        page_confusion_matrices(pdf)
        print("  [8/9] Matrices de confusion ✓")

        page_conclusion(pdf)
        print("  [9/9] Conclusion ✓")

    print(f"\n  ✅ Rapport PDF généré → {PDF_PATH}")
    return PDF_PATH


if __name__ == '__main__':
    generate_pdf()
