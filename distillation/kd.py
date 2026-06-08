"""
distillation/kd.py — Knowledge Distillation.

Formule combinée (Hinton et al., 2015) :
    L = (1 - α) · CE(ŷ_student, y) + α · τ² · KL(σ(z_s/τ) || σ(z_t/τ))

Paramètres fixés : α = 0.5, τ = 4
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple


# ─── Perte de distillation ────────────────────────────────────────────────────
class DistillationLoss(nn.Module):
    """
    Perte combinée CE + KL pour la distillation de connaissances.

    Args:
        temperature : τ — adoucit les distributions (τ > 1 → plus doux)
        alpha       : pondération KL vs CE (0 = CE pur, 1 = KL pur)
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(self,
                z_student: torch.Tensor,
                z_teacher: torch.Tensor,
                labels:    torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Args:
            z_student : (B, C) logits de l'élève
            z_teacher : (B, C) logits de l'enseignant (détachés)
            labels    : (B,)   vraies étiquettes
        Returns:
            (loss_total, loss_ce_scalar, loss_kl_scalar)
        """
        # ① Perte CE standard
        loss_ce = F.cross_entropy(z_student, labels)

        # ② KL-Divergence avec softmax lissé par τ
        log_ps = F.log_softmax(z_student / self.T, dim=1)
        pt     = F.softmax(z_teacher.detach() / self.T, dim=1)
        loss_kl = F.kl_div(log_ps, pt, reduction='batchmean') * (self.T ** 2)

        # ③ Combinaison pondérée
        loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kl
        return loss, loss_ce.item(), loss_kl.item()


# ─── Entraînement par distillation ───────────────────────────────────────────
class DistillationTrainer:
    """
    Entraîne un modèle élève guidé par un enseignant pré-entraîné.
    L'enseignant est figé (mode eval, pas de gradient).
    """

    def __init__(self,
                 teacher:      nn.Module,
                 student:      nn.Module,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 temperature:  float = 4.0,
                 alpha:        float = 0.5,
                 lr:           float = 1e-3,
                 n_epochs:     int   = 25,
                 patience:     int   = 5,
                 device:       torch.device = torch.device('cpu')):
        self.teacher  = teacher.to(device).eval()
        self.student  = student.to(device)
        self.device   = device
        self.n_epochs = n_epochs
        self.patience = patience

        self.criterion = DistillationLoss(temperature, alpha)
        self.optimizer = torch.optim.Adam(
            student.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs)

        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.history = {
            'epoch': [], 'train_loss': [], 'train_ce': [], 'train_kl': [],
            'val_loss': [], 'student_val_acc': []
        }

    def _train_epoch(self) -> Tuple[float, float, float]:
        self.student.train()
        total_loss, total_ce, total_kl = 0.0, 0.0, 0.0
        n = 0

        for pts, labels in self.train_loader:
            pts, labels = pts.to(self.device), labels.to(self.device)
            with torch.no_grad():
                z_t = self.teacher(pts)
            z_s = self.student(pts)
            loss, ce, kl = self.criterion(z_s, z_t, labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_ce   += ce * bs
            total_kl   += kl * bs
            n          += bs

        return total_loss / n, total_ce / n, total_kl / n

    @torch.no_grad()
    def _eval_epoch(self) -> Tuple[float, float]:
        self.student.eval()
        total_loss, correct, total = 0.0, 0, 0

        for pts, labels in self.val_loader:
            pts, labels = pts.to(self.device), labels.to(self.device)
            z_t = self.teacher(pts)
            z_s = self.student(pts)
            loss, _, _ = self.criterion(z_s, z_t, labels)
            total_loss += loss.item() * labels.size(0)
            correct    += (z_s.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        return total_loss / total, correct / total

    def run(self, teacher_acc: float = None) -> Dict[str, Any]:
        """Lance la distillation complète."""
        T   = self.criterion.T
        alp = self.criterion.alpha

        print(f"\n{'='*60}")
        print(f"  Distillation — τ={T} | α={alp} | Epochs max={self.n_epochs}")
        if teacher_acc is not None:
            print(f"  Teacher Accuracy = {teacher_acc:.4f}")
            self.history['teacher_acc_line'] = teacher_acc
        print(f"{'='*60}")

        best_acc  = -1.0
        best_sd   = None
        no_improve = 0
        t_start   = time.time()

        for epoch in range(1, self.n_epochs + 1):
            tr_loss, tr_ce, tr_kl = self._train_epoch()
            val_loss, val_acc     = self._eval_epoch()
            self.scheduler.step()

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(tr_loss)
            self.history['train_ce'].append(tr_ce)
            self.history['train_kl'].append(tr_kl)
            self.history['val_loss'].append(val_loss)
            self.history['student_val_acc'].append(val_acc)

            if val_acc > best_acc:
                best_acc  = val_acc
                best_sd   = copy.deepcopy(self.student.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.n_epochs} | "
                      f"Loss={tr_loss:.4f} (CE={tr_ce:.4f}, KL={tr_kl:.4f}) | "
                      f"Val Acc={val_acc:.4f}")

            if no_improve >= self.patience:
                print(f"  [EarlyStopping] Arrêt à l'époque {epoch}")
                break

        total_time = time.time() - t_start
        self.history['total_time'] = total_time
        self.history['best_val_acc'] = best_acc

        if best_sd is not None:
            self.student.load_state_dict(best_sd)

        print(f"\n  ✓ Meilleure Val Acc student = {best_acc:.4f}")
        print(f"  ✓ Temps distillation = {total_time/60:.1f} min")
        return self.history
