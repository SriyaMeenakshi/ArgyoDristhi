"""
ArogyaDrishti — training utilities
==================================

Shared helpers for both the per-encoder trainer and the multimodal/fusion
trainer: reproducibility, augmentation (matching the architecture document),
class-weight computation, metrics, early stopping and checkpointing.
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from arogyadrishti.encoders import (
    IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD,
)


# --------------------------------------------------------------------------- #
#  Reproducibility                                                            #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Augmentation (per the architecture doc)                                    #
# --------------------------------------------------------------------------- #
class AddGaussianNoise:
    """Adds Gaussian noise to a tensor image (simulates camera noise)."""
    def __init__(self, sigma: float = 0.1, p: float = 0.5):
        self.sigma, self.p = sigma, p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            x = x + torch.randn_like(x) * self.sigma
        return x


def _norm_for(encoder):
    return (CLIP_MEAN, CLIP_STD) if encoder.norm == "clip" \
        else (IMAGENET_MEAN, IMAGENET_STD)


def build_train_transform(encoder, vflip: float = 0.0):
    """Training augmentation. `vflip` 0.2–0.3 for tongue/nail/palm, else 0."""
    mean, std = _norm_for(encoder)
    size = encoder.input_size
    tfms = [
        T.Resize((size, size)),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if vflip > 0:
        tfms.append(T.RandomVerticalFlip(p=vflip))
    tfms += [
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        AddGaussianNoise(sigma=0.1, p=0.5),
    ]
    return T.Compose(tfms)


def build_eval_transform(encoder):
    """Deterministic transform for validation (no augmentation)."""
    mean, std = _norm_for(encoder)
    size = encoder.input_size
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# Modalities that benefit from vertical flips (orientation-agnostic)
VFLIP_MODALITIES = {"tongue": 0.3, "nail": 0.3, "palm": 0.2}


# --------------------------------------------------------------------------- #
#  Class weights for imbalanced classification                                 #
# --------------------------------------------------------------------------- #
def class_weights_from_counts(counts: list[int]) -> torch.Tensor:
    """Inverse-frequency weights normalised to mean 1.0."""
    counts = torch.tensor(counts, dtype=torch.float32).clamp(min=1)
    w = counts.sum() / (len(counts) * counts)
    return w / w.mean()


# --------------------------------------------------------------------------- #
#  Metrics                                                                    #
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred, num_classes):
    """Per-class precision/recall/F1 + macro averages + accuracy."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)),
        average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(x) for x in f1],
    }


def multilabel_metrics(y_true, y_prob, risk_names, threshold: float = 0.5):
    """Per-risk ROC-AUC + F1 for the 7 risk heads (multi-label)."""
    from sklearn.metrics import roc_auc_score, f1_score
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    aucs, f1s = [], []
    for i, name in enumerate(risk_names):
        col_true = y_true[:, i]
        try:
            auc = roc_auc_score(col_true, y_prob[:, i]) \
                if len(np.unique(col_true)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        f1 = f1_score(col_true, y_pred[:, i], zero_division=0)
        out[name] = {"auc": float(auc), "f1": float(f1)}
        if not np.isnan(auc):
            aucs.append(auc)
        f1s.append(f1)
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["macro_f1"] = float(np.mean(f1s))
    return out


# --------------------------------------------------------------------------- #
#  Early stopping                                                             #
# --------------------------------------------------------------------------- #
class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "max", min_delta=1e-4):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Return True if `value` is a new best."""
        if self.best is None:
            self.best = value
            return True
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# --------------------------------------------------------------------------- #
#  Checkpoints                                                                #
# --------------------------------------------------------------------------- #
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
