"""
smoke_test.py — Test de fumée ultra-rapide pour valider toute la chaîne de classification 3D.
Utilise des données synthétiques minimalistes.
Durée cible : < 5 secondes sur CPU.
"""

import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from dataset import get_dataloaders, SELECTED_CLASSES, NUM_CLASSES
from models import get_model, get_student
from utils import count_parameters
from federated.hfl import fedavg_aggregate, client_train
from federated.vfl import XYZEncoder, RGBEncoder, ServerFusion
from distillation.kd import DistillationLoss

# ─── Config mini ─────────────────────────────────────────────────────────────
N_CLASSES = 8
N_PTS     = 256
BATCH     = 4
N_OBJ     = 16
device    = torch.device('cpu')

print("=" * 55)
print("  SMOKE TEST - Validation du pipeline complet (Nouveau)")
print("=" * 55)

# ─── Données synthétiques ────────────────────────────────────────────────────
def make_data(n=N_OBJ):
    x = torch.randn(n, 6, N_PTS)
    y = torch.randint(0, N_CLASSES, (n,))
    return TensorDataset(x, y)

train_ds   = make_data(N_OBJ)
val_ds     = make_data(N_OBJ // 4)
train_ld   = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_ld     = DataLoader(val_ds,   batch_size=BATCH)

# ─── Test 1 : GIN forward ──────────────────────────────────────────────────
print("\n[1] GIN forward pass...")
gin = get_model('gin', num_classes=N_CLASSES, k=8)
x_t = torch.randn(BATCH, 6, N_PTS)
out = gin(x_t)
assert out.shape == (BATCH, N_CLASSES), f"Mauvais shape GIN: {out.shape}"
print(f"    OK - output: {out.shape} | params: {count_parameters(gin):,}")

# ─── Test 2 : TACO-Net forward ───────────────────────────────────────────────
print("\n[2] TACO-Net forward pass...")
taconet = get_model('taconet', num_classes=N_CLASSES, k=8)
out2 = taconet(x_t)
assert out2.shape == (BATCH, N_CLASSES), f"Mauvais shape TACO-Net: {out2.shape}"
print(f"    OK - output: {out2.shape} | params: {count_parameters(taconet):,}")

# ─── Test 3 : Student models ────────────────────────────────────────────────
print("\n[3] Modèles étudiants (StudentGIN, StudentTACO)...")
s_gin  = get_student('gin', num_classes=N_CLASSES, k=8)
s_taco = get_student('taconet', num_classes=N_CLASSES, k=8)
out3   = s_gin(x_t)
out4   = s_taco(x_t)
assert out3.shape == (BATCH, N_CLASSES), f"Mauvais shape StudentGIN: {out3.shape}"
assert out4.shape == (BATCH, N_CLASSES), f"Mauvais shape StudentTACO: {out4.shape}"
ratio_g = count_parameters(gin) / count_parameters(s_gin)
ratio_t = count_parameters(taconet) / count_parameters(s_taco)
print(f"    OK - StudentGIN  params: {count_parameters(s_gin):,} (ratio: {ratio_g:.1f}x)")
print(f"    OK - StudentTACO params: {count_parameters(s_taco):,} (ratio: {ratio_t:.1f}x)")

# ─── Test 4 : FedAvg aggregation ─────────────────────────────────────────────
print("\n[4] FedAvg aggregation manuelle...")
import copy
global_model = get_model('gin', num_classes=N_CLASSES, k=8)
client_updates = []
for i in range(3):
    local = copy.deepcopy(global_model)
    for p in local.parameters():
        p.data += torch.randn_like(p) * 0.01
    client_updates.append((copy.deepcopy(local.state_dict()), 100 + i * 20))

orig_p = list(copy.deepcopy(global_model).parameters())[0].data.clone()
agg_model = fedavg_aggregate(global_model, client_updates)
agg_p  = list(agg_model.parameters())[0].data
assert not torch.allclose(orig_p, agg_p), "Les poids n'ont pas changé!"
print(f"    OK - Agrégation FedAvg sur 3 clients réussie")

# ─── Test 5 : Entraînement local client ──────────────────────────────────────
print("\n[5] Entraînement local client (1 epoch)...")
local_model = copy.deepcopy(global_model)
sd, n, loss = client_train(
    local_model, train_ld, lr=1e-3, local_epochs=1, device=device)
assert isinstance(sd, dict), "state_dict invalide"
assert n == len(train_ds)
print(f"    OK - state_dict de {len(sd)} clés | {n} échantillons | loss={loss:.4f}")

# ─── Test 6 : Distillation Loss ──────────────────────────────────────────────
print("\n[6] Distillation Loss (CE + KL)...")
loss_fn   = DistillationLoss(temperature=4.0, alpha=0.5)
z_teacher = torch.randn(BATCH, N_CLASSES)
z_student = torch.randn(BATCH, N_CLASSES, requires_grad=True)
labels    = torch.randint(0, N_CLASSES, (BATCH,))
loss, ce, kl = loss_fn(z_student, z_teacher, labels)
loss.backward()
assert z_student.grad is not None, "Gradient non calculé!"
assert z_teacher.grad is None, "Teacher ne doit pas avoir de gradient!"
print(f"    OK - Loss={loss.item():.4f} | CE={ce:.4f} | KL={kl:.4f}")

# ─── Test 7 : VFL forward ────────────────────────────────────────────────────
print("\n[7] VFL (XYZEncoder, RGBEncoder, ServerFusion)...")
enc_a  = XYZEncoder(embed_dim=64).to(device)
enc_b  = RGBEncoder(embed_dim=64).to(device)
server = ServerFusion(embed_dim=64, num_classes=N_CLASSES).to(device)
xyz    = x_t[:, :3, :]
rgb    = x_t[:, 3:, :]
e_A    = enc_a(xyz)
e_B    = enc_b(rgb)
logits = server(e_A, e_B)
assert logits.shape == (BATCH, N_CLASSES), f"Mauvais shape VFL logits: {logits.shape}"
print(f"    OK - e_A: {e_A.shape} | e_B: {e_B.shape} | logits: {logits.shape}")

# ─── Bilan ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  [OK] TOUS LES TESTS PASSÉS (7/7)")
print("=" * 55)
print()
