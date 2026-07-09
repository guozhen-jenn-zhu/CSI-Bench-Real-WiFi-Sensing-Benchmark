"""3-phase trainer for prunedAttentionGRU on the CSI-Bench supervised pipeline.

Phase 1 (pretrain): standard training with optional mixup, cosine LR.
Phase 2 (prune):    one-shot std-based pruning to a target sparsity.
Phase 3 (finetune): mixup off, lower LR, mask kept frozen by MaskedLinear's
                    forward (weight * mask).

Reports dense / pruned-zero-shot / pruned+finetuned accuracy & F1 to the
results directory. Compatible with TaskTrainer's I/O conventions: the model
exposes (B, num_classes) logits; loaders yield (inputs, label_idx).
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def _mixup_batch(x, y_idx, num_classes, alpha):
    if alpha <= 0:
        return x, y_idx, y_idx, 1.0, False
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y_idx, y_idx[idx], lam, True


def _evaluate(model, loader, device, num_classes):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    n = 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            if isinstance(y, tuple):
                y = y[0]
            if x.size(0) == 0:
                continue
            x = x.to(device)
            y = y.to(device).long()
            logits = model(x)
            total_loss += ce(logits, y).item()
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.cpu().numpy())
            n += x.size(0)
    if n == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "f1_macro": 0.0}
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return {
        "loss": total_loss / n,
        "accuracy": float(accuracy_score(targets, preds)),
        "f1_macro": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def _train_one_epoch(model, loader, optimizer, criterion, device, num_classes,
                     mixup_alpha=0.0, mixup_prob=0.0, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        if isinstance(y, tuple):
            y = y[0]
        if x.size(0) == 0:
            continue
        x = x.to(device)
        y = y.to(device).long()
        bs = x.size(0)

        if mixup_prob > 0 and np.random.rand() < mixup_prob:
            x, y_a, y_b, lam, mixed = _mixup_batch(x, y, num_classes, mixup_alpha)
        else:
            y_a, y_b, lam, mixed = y, y, 1.0, False

        optimizer.zero_grad()
        logits = model(x)
        if mixed:
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            loss = criterion(logits, y)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item() * bs
        correct += int((logits.argmax(dim=1) == y).sum().item())
        n += bs
    if n == 0:
        return 0.0, 0.0
    return total_loss / n, correct / n


class PAGTrainer:
    """3-phase trainer for PrunedAttentionGRUClassifier."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loaders,
        num_classes,
        save_path,
        device=None,
        config=None,
        label_mapper=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loaders = test_loaders or {}
        self.num_classes = num_classes
        self.save_path = save_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        self.label_mapper = label_mapper
        os.makedirs(save_path, exist_ok=True)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Hyperparameters with sensible defaults for the unified workflow.
        self.epochs_pre = int(self.config.get("epochs_pretrain", 80))
        self.epochs_ft = int(self.config.get("epochs_finetune", 20))
        self.lr = float(self.config.get("learning_rate", 1e-3))
        self.lr_ft = float(self.config.get("learning_rate_finetune", self.lr * 0.1))
        self.weight_decay = float(self.config.get("weight_decay", 1e-5))
        self.mixup_alpha = float(self.config.get("mixup_alpha", 1.0))
        self.mixup_prob = float(self.config.get("mixup_prob", 0.7))
        # Default target sparsity = 80% zeros (k = 0.20 = fraction of weights kept).
        self.prune_method = self.config.get("prune_method", "std")
        self.prune_s = float(self.config.get("prune_s", 0.5))
        self.prune_k = float(self.config.get("prune_k", 0.20))
        self.prune_connectivity = float(self.config.get("prune_connectivity", 0.20))

    def _make_optimizer(self, params, lr, epochs):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        return opt, sched

    def _train_phase(self, name, epochs, lr, mixup_prob, mixup_alpha):
        if epochs <= 0:
            return [], None
        optimizer, scheduler = self._make_optimizer(self.model.parameters(), lr, epochs)
        history = []
        best_val_acc = -1.0
        best_state = None
        for epoch in range(epochs):
            t0 = time.time()
            tl, ta = _train_one_epoch(
                self.model, self.train_loader, optimizer, self.criterion,
                self.device, self.num_classes,
                mixup_alpha=mixup_alpha, mixup_prob=mixup_prob,
            )
            scheduler.step()
            val_metrics = _evaluate(self.model, self.val_loader, self.device, self.num_classes)
            row = {
                "phase": name,
                "epoch": epoch + 1,
                "train_loss": tl,
                "train_acc": ta,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1_macro"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_s": time.time() - t0,
            }
            history.append(row)
            print(
                f"[{name}] epoch {epoch+1}/{epochs} "
                f"train_loss={tl:.4f} train_acc={ta:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1_macro']:.4f}"
            )
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_state = copy.deepcopy(self.model.state_dict())
        return history, best_state

    def _eval_all_test_splits(self):
        out = {}
        for name, loader in self.test_loaders.items():
            out[name] = _evaluate(self.model, loader, self.device, self.num_classes)
        return out

    def train(self):
        results = {
            "phases": {},
            "test": {"dense": {}, "pruned_zero_shot": {}, "pruned_finetuned": {}},
            "sparsity": {},
            "config": {
                "epochs_pretrain": self.epochs_pre,
                "epochs_finetune": self.epochs_ft,
                "lr": self.lr,
                "lr_finetune": self.lr_ft,
                "weight_decay": self.weight_decay,
                "mixup_alpha": self.mixup_alpha,
                "mixup_prob": self.mixup_prob,
                "prune_method": self.prune_method,
                "prune_s": self.prune_s,
                "prune_k": self.prune_k,
                "prune_connectivity": self.prune_connectivity,
            },
        }

        # Phase 1: pretrain with mixup.
        print("=== Phase 1: pretrain ===")
        hist_pre, best_dense_state = self._train_phase(
            "pretrain", self.epochs_pre, self.lr, self.mixup_prob, self.mixup_alpha,
        )
        results["phases"]["pretrain"] = hist_pre
        if best_dense_state is not None:
            self.model.load_state_dict(best_dense_state)
        results["test"]["dense"] = self._eval_all_test_splits()
        results["sparsity"]["dense"] = self.model.sparsity_report() \
            if hasattr(self.model, "sparsity_report") else {}
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_path, "pag_dense.pt"))

        # Phase 2: one-shot prune.
        print("=== Phase 2: prune ===")
        if hasattr(self.model, "apply_pruning"):
            self.model.apply_pruning(
                method=self.prune_method,
                s=self.prune_s,
                k=self.prune_k,
                connectivity=self.prune_connectivity,
            )
        results["test"]["pruned_zero_shot"] = self._eval_all_test_splits()
        results["sparsity"]["pruned"] = self.model.sparsity_report() \
            if hasattr(self.model, "sparsity_report") else {}
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_path, "pag_pruned.pt"))

        # Phase 3: finetune with mixup off.
        print("=== Phase 3: finetune ===")
        hist_ft, best_ft_state = self._train_phase(
            "finetune", self.epochs_ft, self.lr_ft, mixup_prob=0.0, mixup_alpha=0.0,
        )
        results["phases"]["finetune"] = hist_ft
        if best_ft_state is not None:
            self.model.load_state_dict(best_ft_state)
        results["test"]["pruned_finetuned"] = self._eval_all_test_splits()
        results["sparsity"]["pruned_finetuned"] = self.model.sparsity_report() \
            if hasattr(self.model, "sparsity_report") else {}
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_path, "pag_pruned_finetuned.pt"))

        with open(os.path.join(self.save_path, "pag_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return self.model, results
