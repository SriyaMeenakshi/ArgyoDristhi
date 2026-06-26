"""
ArogyaDrishti — Stage 2: multimodal / fusion trainer
====================================================

Trains the cross-modal attention fusion + 7 risk heads (and the tabular
encoder, and optionally fine-tunes the image encoders) with a MULTI-TASK loss.

Because no single source has every modality for every person, training is
driven by a MANIFEST CSV with one row per person.  Any cell may be empty —
missing modalities are handled per-person via the fusion mask.

Manifest columns
----------------
  image paths (any may be empty):
      eye_path, face_path, tongue_path, skin_path, nail_path, palm_path
  21 tabular features (any may be empty -> filled with population mean;
      if ALL empty the tabular modality is treated as absent):
      age, gender, bmi, systolic_bp, diastolic_bp, heart_rate, spo2,
      hemoglobin, glucose_fasting, glucose_pp, hba1c, alt, ast,
      bilirubin_total, creatinine, urea, ferritin, vitamin_b12, vitamin_d,
      albumin, fatigue_score
  7 risk labels (0/1 or 0.0–1.0):
      risk_hematological, risk_metabolic, risk_renal, risk_hepatic,
      risk_cardiovascular, risk_dermatological, risk_nutritional

Usage
-----
    python -m arogyadrishti.training.train_multimodal \
        --manifest data/multimodal.csv --epochs 50 --batch-size 16 \
        --ckpt-dir checkpoints --out-dir checkpoints

    # validate the whole pipeline end-to-end with synthetic data:
    python -m arogyadrishti.training.train_multimodal --smoke-test
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from arogyadrishti.system import ArogyaDrishti, _IMAGE_ENCODERS
from arogyadrishti.fusion import RISK_CATEGORIES
from arogyadrishti.preprocessing import (
    preprocess_tabular, FEATURE_ORDER, load_image,
)
from arogyadrishti.training.train_utils import (
    set_seed, build_train_transform, build_eval_transform, VFLIP_MODALITIES,
    multilabel_metrics, EarlyStopping, save_checkpoint, AverageMeter,
)

IMAGE_MODS = list(_IMAGE_ENCODERS)            # eye, face, tongue, skin, nail, palm
RISK_COLS = [f"risk_{c}" for c in RISK_CATEGORIES]


# --------------------------------------------------------------------------- #
#  Dataset                                                                    #
# --------------------------------------------------------------------------- #
class MultimodalManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: dict, train: bool):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms      # {modality: torchvision transform}
        self.train = train
        # cache a blank tensor per modality for absent images
        self.blanks = {
            m: torch.zeros(3, t.transforms[0].size[0], t.transforms[0].size[0])
            for m, t in transforms.items()
        }

    def __len__(self):
        return len(self.df)

    def _load_img(self, path, modality):
        if isinstance(path, str) and path and os.path.exists(path):
            img = load_image(path)
            return self.transforms[modality](img), True
        return self.blanks[modality].clone(), False

    def __getitem__(self, i):
        row = self.df.iloc[i]
        images, img_present = {}, {}
        for m in IMAGE_MODS:
            t, present = self._load_img(row.get(f"{m}_path", ""), m)
            images[m] = t
            img_present[m] = present

        # tabular
        feats = {k: row[k] for k in FEATURE_ORDER
                 if k in row and pd.notna(row[k])}
        tab_present = len(feats) > 0
        tabular = preprocess_tabular(feats).squeeze(0)      # (21,)

        labels = torch.tensor(
            [float(row[c]) for c in RISK_COLS], dtype=torch.float32)

        return {
            "images": images, "img_present": img_present,
            "tabular": tabular, "tab_present": tab_present,
            "labels": labels,
        }


def collate(batch):
    out = {"images": {}, "img_present": {}}
    for m in IMAGE_MODS:
        out["images"][m] = torch.stack([b["images"][m] for b in batch])
        out["img_present"][m] = torch.tensor(
            [b["img_present"][m] for b in batch], dtype=torch.bool)
    out["tabular"] = torch.stack([b["tabular"] for b in batch])
    out["tab_present"] = torch.tensor(
        [b["tab_present"] for b in batch], dtype=torch.bool)
    out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# --------------------------------------------------------------------------- #
#  Train / eval                                                               #
# --------------------------------------------------------------------------- #
def compute_embeddings(system, batch, device):
    """Build {modality: emb} and {modality: present-mask} for the batch.
    A modality is only encoded if at least one sample in the batch has it."""
    embeddings, masks = {}, {}
    for m in IMAGE_MODS:
        present = batch["img_present"][m].to(device)
        if present.any():
            x = batch["images"][m].to(device)
            embeddings[m] = system.image_encoders[m].get_embedding(x)
            masks[m] = present
    if batch["tab_present"].any():
        x = batch["tabular"].to(device)
        embeddings["tabular"] = system.tabular_encoder.get_embedding(x)
        masks["tabular"] = batch["tab_present"].to(device)
    return embeddings, masks


def run_epoch(system, loader, criterion, device, optimizer=None,
              freeze_encoders=True, clip=1.0):
    train = optimizer is not None
    # fusion + tabular train/eval; image encoders stay eval when frozen
    system.fusion.train(train)
    system.tabular_encoder.train(train)
    for enc in system.image_encoders.values():
        enc.train(train and not freeze_encoders)

    loss_meter = AverageMeter()
    all_true, all_prob = [], []

    for batch in loader:
        labels = batch["labels"].to(device)
        with torch.set_grad_enabled(train):
            embeddings, masks = compute_embeddings(system, batch, device)
            out = system.fusion(embeddings, masks)
            loss = criterion(out["risk_logits"], labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in system.parameters() if p.requires_grad], clip)
                optimizer.step()
        loss_meter.update(loss.item(), labels.size(0))
        all_true.extend(labels.detach().cpu().tolist())
        all_prob.extend(out["risks"].detach().cpu().tolist())

    metrics = multilabel_metrics(all_true, all_prob, RISK_CATEGORIES)
    metrics["loss"] = loss_meter.avg
    return metrics


def pos_weight_from_labels(df) -> torch.Tensor:
    """pos_weight = neg/pos per risk, to counter class imbalance in BCE."""
    y = df[RISK_COLS].values.astype(float)
    pos = y.sum(0).clip(min=1)
    neg = (1 - y).sum(0).clip(min=1)
    return torch.tensor(neg / pos, dtype=torch.float32)


def load_stage1_checkpoints(system, ckpt_dir):
    """Load per-encoder weights produced by train_image_encoder (if present)."""
    for m, enc in system.image_encoders.items():
        path = os.path.join(ckpt_dir, f"{m}_best.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            # only the backbone/shared weights are reused; the per-modality
            # classifier head shape may differ, so load non-strictly.
            enc.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"   loaded stage-1 weights for {m} from {path}")


# --------------------------------------------------------------------------- #
#  Synthetic data for --smoke-test                                            #
# --------------------------------------------------------------------------- #
def make_smoke_manifest(tmpdir, n=24):
    os.makedirs(tmpdir, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = {}
        for m in IMAGE_MODS:
            # ~70% of images present; tongue needs a larger image
            if rng.random() < 0.7:
                p = os.path.join(tmpdir, f"{m}_{i}.jpg")
                Image.fromarray(
                    (rng.random((320, 320, 3)) * 255).astype("uint8")).save(p)
                row[f"{m}_path"] = p
            else:
                row[f"{m}_path"] = ""
        # tabular present ~80% of the time
        if rng.random() < 0.8:
            for k in FEATURE_ORDER:
                row[k] = rng.normal(0, 1)
            row["gender"] = rng.integers(0, 2)
        for c in RISK_COLS:
            row[c] = int(rng.random() < 0.4)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(tmpdir, "manifest.csv")
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--val-manifest", help="optional separate val manifest")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-dir", default=None,
                    help="dir with stage-1 {modality}_best.pt to warm-start")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--finetune-encoders", action="store_true",
                    help="also update image encoders (default: frozen)")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # smoke test synthesises data and forces offline random weights
    if args.smoke_test:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="arogya_smoke_")
        args.manifest = make_smoke_manifest(tmp)
        args.pretrained = False
        args.epochs = 1
        print(f"[smoke-test] synthetic manifest at {args.manifest}")

    system = ArogyaDrishti(pretrained=args.pretrained, device=str(device))
    if args.ckpt_dir:
        load_stage1_checkpoints(system, args.ckpt_dir)

    freeze = not args.finetune_encoders
    if freeze:
        for enc in system.image_encoders.values():
            for p in enc.parameters():
                p.requires_grad = False

    # transforms per modality, built from the encoders that actually loaded
    train_tfms = {m: build_train_transform(system.image_encoders[m],
                                            VFLIP_MODALITIES.get(m, 0.0))
                  for m in IMAGE_MODS}
    eval_tfms = {m: build_eval_transform(system.image_encoders[m])
                 for m in IMAGE_MODS}

    df = pd.read_csv(args.manifest)
    if args.val_manifest:
        train_df, val_df = df, pd.read_csv(args.val_manifest)
    else:
        val_n = max(1, int(len(df) * args.val_frac))
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        val_df, train_df = df.iloc[:val_n], df.iloc[val_n:]

    train_ds = MultimodalManifestDataset(train_df, train_tfms, train=True)
    val_ds = MultimodalManifestDataset(val_df, eval_tfms, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    pos_weight = pos_weight_from_labels(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # multi-task loss

    params = [p for p in system.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    stopper = EarlyStopping(patience=args.patience, mode="max")
    best_path = f"{args.out_dir}/fusion_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(system, train_loader, criterion, device, optimizer,
                       freeze_encoders=freeze)
        va = run_epoch(system, val_loader, criterion, device,
                       freeze_encoders=freeze)
        scheduler.step()
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train loss {tr['loss']:.4f} F1 {tr['macro_f1']:.3f} | "
              f"val loss {va['loss']:.4f} F1 {va['macro_f1']:.3f} "
              f"AUC {va['macro_auc']:.3f}")

        score = va["macro_f1"] if np.isnan(va["macro_auc"]) else va["macro_auc"]
        if stopper.step(score):
            save_checkpoint({
                "fusion": system.fusion.state_dict(),
                "tabular_encoder": system.tabular_encoder.state_dict(),
                "image_encoders": {m: e.state_dict()
                                   for m, e in system.image_encoders.items()},
                "val_metrics": va, "epoch": epoch,
            }, best_path)
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (best={stopper.best:.3f})")
            break

    print(f"Done. Best checkpoint -> {best_path}")
    if args.smoke_test:
        print("✅ smoke-test passed: data loading, multi-task loss and "
              "train/val loop all run.")


if __name__ == "__main__":
    main()
