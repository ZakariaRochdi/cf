"""
dataset.py — Pipeline de données pour la classification 3D.
Sous-ensemble de 8 classes ShapeNet, 512 points par objet, batch=8.
Génération synthétique automatique si les .ply ne sont pas disponibles.

Classes : airplane, car, chair, table, lamp, bottle, bed, sofa
Split   : Train 70% / Val 15% / Test 15%  (seed=42, reproductible)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ─── Seed globale (reproductibilité) ─────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── 8 classes retenues ───────────────────────────────────────────────────────
SELECTED_CLASSES = ['airplane', 'car', 'chair', 'table', 'lamp', 'bottle', 'bed', 'sofa']
CLASS_TO_IDX     = {c: i for i, c in enumerate(SELECTED_CLASSES)}
IDX_TO_CLASS     = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES      = len(SELECTED_CLASSES)   # 8

# ─── Paramètres par défaut ────────────────────────────────────────────────────
NUM_POINTS       = 512    # sous-échantillonnage fixe
MAX_PER_CLASS    = 500    # max 500 objets par classe (CPU raisonnable)
BATCH_SIZE       = 8      # batch size CPU-optimisé

# ─── Partition fédérée horizontale (non-IID) ─────────────────────────────────
# 4 clients avec partitions sémantiques distinctes
HFL_PARTITION = {
    0: ['airplane', 'car'],           # Client 0 — Transport
    1: ['chair', 'table'],            # Client 1 — Mobilier plat
    2: ['lamp', 'bottle'],            # Client 2 — Objets cylindriques
    3: ['bed', 'sofa'],               # Client 3 — Mobilier lounge
}
CLIENT_NAMES = ['Transport', 'Mobilier', 'Cylindrique', 'Lounge']


# ─── Générateurs synthétiques ─────────────────────────────────────────────────
def _sample_sphere(n: int) -> np.ndarray:
    pts = np.random.randn(n, 3).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
    return pts


def _sample_cylinder(n: int) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi, n)
    h     = np.random.uniform(-1, 1, n)
    return np.stack([np.cos(theta), np.sin(theta), h], axis=1).astype(np.float32)


def _sample_box(n: int) -> np.ndarray:
    face = np.random.randint(0, 6, n)
    pts  = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
    for i, f in enumerate(face):
        ax = f // 2
        pts[i, ax] = 1.0 if f % 2 == 0 else -1.0
    return pts


def _sample_elongated(n: int, ratio: float = 3.0) -> np.ndarray:
    pts = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
    pts[:, 0] *= ratio
    return pts


# Mapping classe → forme géométrique caractéristique
_CLASS_GENERATOR = {
    'airplane': lambda n: _sample_elongated(n, 3.5),
    'car':      lambda n: _sample_box(n) * np.array([1.8, 0.7, 0.8], dtype=np.float32),
    'chair':    lambda n: _sample_box(n) * np.array([0.6, 1.0, 0.6], dtype=np.float32),
    'table':    lambda n: _sample_box(n) * np.array([1.5, 0.8, 1.2], dtype=np.float32),
    'lamp':     lambda n: _sample_cylinder(n) * np.array([0.4, 0.4, 1.5], dtype=np.float32),
    'bottle':   lambda n: _sample_cylinder(n) * np.array([0.3, 0.3, 1.2], dtype=np.float32),
    'bed':      lambda n: _sample_box(n) * np.array([1.5, 0.4, 1.0], dtype=np.float32),
    'sofa':     lambda n: _sample_box(n) * np.array([1.5, 0.9, 0.7], dtype=np.float32),
}


def generate_point_cloud(label: str, num_points: int = NUM_POINTS) -> np.ndarray:
    """
    Génère un nuage de points synthétique pour la classe donnée.
    Retourne (num_points, 6) : [x, y, z, r, g, b] normalisés.
    """
    gen = _CLASS_GENERATOR.get(label, _sample_sphere)
    xyz = gen(num_points).astype(np.float32)
    # Bruit gaussien léger
    xyz += np.random.normal(0, 0.02, xyz.shape).astype(np.float32)
    # Normalisation dans la sphère unité
    centroid = xyz.mean(axis=0)
    xyz -= centroid
    scale = np.max(np.linalg.norm(xyz, axis=1)) + 1e-8
    xyz /= scale
    # Couleurs synthétiques basées sur la position (simuler la texture)
    rgb = np.clip((xyz + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    return np.concatenate([xyz, rgb], axis=1)   # (N, 6)


# ─── Augmentations (train only) ───────────────────────────────────────────────
def augment_point_cloud(pts: np.ndarray) -> np.ndarray:
    """
    Augmentations légères : rotation Z, jitter faible, scaling léger.
    """
    xyz = pts[:, :3].copy()
    rgb = pts[:, 3:].copy()
    # Rotation aléatoire autour de Z
    angle = np.random.uniform(0, 2 * np.pi)
    c, s  = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    xyz = xyz @ R.T
    # Jitter faible
    xyz += np.random.normal(0, 0.01, xyz.shape).astype(np.float32)
    # Scaling léger [0.9, 1.1]
    xyz *= np.random.uniform(0.9, 1.1)
    return np.concatenate([xyz, rgb], axis=1)


# ─── Dataset PyTorch ──────────────────────────────────────────────────────────
class PointCloudDataset(Dataset):
    """
    Dataset de nuages de points 3D pour les 8 classes sélectionnées.
    Génération synthétique si .ply non disponibles.
    """

    def __init__(self, records: pd.DataFrame, ply_dir: str = None,
                 num_points: int = NUM_POINTS, augment: bool = False):
        self.records    = records.reset_index(drop=True)
        self.ply_dir    = ply_dir
        self.num_points = num_points
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.records)

    def _load_ply(self, filepath: str):
        try:
            import open3d as o3d
            pcd  = o3d.io.read_point_cloud(filepath)
            pts  = np.asarray(pcd.points, dtype=np.float32)
            cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() \
                   else np.zeros((len(pts), 3), dtype=np.float32)
            return np.concatenate([pts, cols], axis=1)
        except Exception:
            return None

    def _subsample(self, pts: np.ndarray) -> np.ndarray:
        n = len(pts)
        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.random.choice(n, self.num_points, replace=True)
        return pts[idx]

    def __getitem__(self, idx):
        row   = self.records.iloc[idx]
        label = row['label']
        lid   = CLASS_TO_IDX[label]

        pts = None
        if self.ply_dir:
            path = os.path.join(self.ply_dir, row['id'] + '.ply')
            if os.path.exists(path):
                pts = self._load_ply(path)

        if pts is None:
            pts = generate_point_cloud(label, self.num_points)
        else:
            pts = self._subsample(pts)
            centroid    = pts[:, :3].mean(axis=0)
            pts[:, :3] -= centroid
            scale       = np.max(np.linalg.norm(pts[:, :3], axis=1)) + 1e-8
            pts[:, :3] /= scale

        if self.augment:
            pts = augment_point_cloud(pts)

        # Retourne (6, N) pour convolutions Conv1D/2D
        return (torch.tensor(pts.T, dtype=torch.float32),
                torch.tensor(lid, dtype=torch.long))


# ─── Construction des splits ──────────────────────────────────────────────────
def build_splits(csv_path: str, ply_dir: str = None,
                 max_per_class: int = MAX_PER_CLASS,
                 num_points: int = NUM_POINTS,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15):
    """
    Construit les DataLoaders train/val/test.
    Filtre les 8 classes sélectionnées, équilibre max_per_class exemples par classe.
    Split stratifié reproductible (seed=42).
    """
    df = pd.read_csv(csv_path)
    # Garder seulement les 8 classes
    df = df[df['label'].isin(SELECTED_CLASSES)].copy()

    # Sous-ensemble équilibré
    groups = []
    for cls in SELECTED_CLASSES:
        sub = df[df['label'] == cls]
        n   = min(len(sub), max_per_class)
        if n > 0:
            groups.append(sub.sample(n, random_state=SEED))
        else:
            # Créer des entrées synthétiques si classe absente du CSV
            synthetic = pd.DataFrame([{'id': f'synthetic_{cls}_{i}', 'label': cls}
                                       for i in range(max_per_class)])
            groups.append(synthetic)

    df = pd.concat(groups).reset_index(drop=True)

    # Split stratifié 70/15/15
    train_df, temp_df = train_test_split(
        df, test_size=val_ratio + test_ratio,
        stratify=df['label'], random_state=SEED)
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df['label'], random_state=SEED)

    print(f"  Dataset — {NUM_CLASSES} classes | "
          f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    for cls in SELECTED_CLASSES:
        n = (df['label'] == cls).sum()
        print(f"    {cls:<10}: {n} exemples")

    train_ds = PointCloudDataset(train_df, ply_dir, num_points, augment=True)
    val_ds   = PointCloudDataset(val_df,   ply_dir, num_points, augment=False)
    test_ds  = PointCloudDataset(test_df,  ply_dir, num_points, augment=False)

    return train_ds, val_ds, test_ds, train_df, val_df, test_df


def get_dataloaders(csv_path: str, ply_dir: str = None,
                    max_per_class: int = MAX_PER_CLASS,
                    num_points: int = NUM_POINTS,
                    batch_size: int = BATCH_SIZE):
    """Retourne les DataLoaders train/val/test et les DataFrames."""
    train_ds, val_ds, test_ds, train_df, val_df, test_df = build_splits(
        csv_path, ply_dir, max_per_class, num_points)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader, train_df, val_df, test_df


def build_hfl_clients(train_df: pd.DataFrame, ply_dir: str = None,
                      num_points: int = NUM_POINTS,
                      batch_size: int = BATCH_SIZE,
                      iid: bool = False):
    """
    Construit les DataLoaders pour les 4 clients fédérés horizontaux.
    iid=False : partition sémantique (non-IID)
    iid=True  : mélange aléatoire équilibré (IID)
    """
    client_loaders = []
    if iid:
        shuffled = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        splits   = np.array_split(shuffled, 4)
        for sdf in splits:
            ds = PointCloudDataset(sdf, ply_dir, num_points, augment=True)
            client_loaders.append(
                DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0))
    else:
        for cid, classes in HFL_PARTITION.items():
            sdf = train_df[train_df['label'].isin(classes)]
            ds  = PointCloudDataset(sdf, ply_dir, num_points, augment=True)
            client_loaders.append(
                DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0))
    return client_loaders
