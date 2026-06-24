"""
ArogyaDrishti — Stage 1: per-encoder image trainer
===================================================

Trains ONE image encoder + its classification head on its OWN dataset (the
separate per-modality databases).  This is the realistic first stage: each of
the six image datasets has its own classes, so each encoder is fine-tuned
independently before the fusion stage.

Expected data layout (standard ImageFolder):

    data_dir/
        <class_0>/  img1.jpg  img2.jpg ...
        <class_1>/  ...
        ...

The split is done BEFORE augmentation (per the architecture doc) to avoid
leakage: real images are split 80/20, then only the TRAIN subset is augmented;
the VAL subset uses deterministic transforms.

Usage
-----
    python -m arogyadrishti.training.train_image_encoder \
        --modality eye --data-dir /path/to/eyes_defy_anemia \
        --epochs 50 --batch-size 32 --out-dir checkpoints

Run `--help` for all options.  `--smoke-test` runs 1 epoch on whatever tiny
dataset you point it at, to verify the pipeline.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

from arogyadrishti.encoders import (
    EyeEncoder, FaceEncoder, TongueEncoder, SkinEncoder,
    NailEncoder, PalmEncoder,
)
from arogyadrishti.training.train_utils import (
    set_seed, build_train_transform, build_eval_transform, VFLIP_MODALITIES,
    class_weights_from_counts, classification_metrics, EarlyStopping,
    save_checkpoint, AverageMeter,
)

ENCODER_CLASSES = {
    "eye": EyeEncoder, "face": FaceEncoder, "tongue": TongueEncoder,
    "skin": SkinEncoder, "nail": NailEncoder, "palm": PalmEncoder,
}


def build_loaders(data_dir, encoder, batch_size, val_frac, seed, num_workers):
    """Split-before-augment ImageFolder loaders."""
    # Two views of the same folder: one augmented (train), one clean (val).
    vflip = VFLIP_MODALITIES.get(encoder.name, 0.0)
    train_view = ImageFolder(data_dir, transform=build_train_transform(encoder, vflip))
    val_view = ImageFolder(data_dir, transform=build_eval_transform(encoder))

    targets = [y for _, y in train_view.samples]
    idx = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_frac, stratify=targets, random_state=seed)

    train_ds = Subset(train_view, train_idx)
    val_ds = Subset(val_view, val_idx)

    # class counts on the TRAIN split only (for class weights)
    counts = np.bincount([targets[i] for i in train_idx],
                         minlength=len(train_view.classes)).tolist()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader, train_view.classes, counts


def run_epoch(model, loader, criterion, device, optimizer=None, clip=1.0):
    train = optimizer is not None
    model.train(train)
    loss_meter = AverageMeter()
    all_true, all_pred = [], []

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)                      # classification logits
        loss = criterion(logits, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        loss_meter.update(loss.item(), images.size(0))
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(logits.argmax(1).cpu().tolist())
    torch.set_grad_enabled(True)

    metrics = classification_metrics(all_true, all_pred, model.num_classes)
    metrics["loss"] = loss_meter.avg
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=list(ENCODER_CLASSES))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run a single epoch to validate the pipeline")
    ap.add_argument("--resume", action="store_true",
                    help="resume training from existing checkpoint")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # peek at the number of classes from the folder
    n_classes = len(ImageFolder(args.data_dir).classes)
    enc_cls = ENCODER_CLASSES[args.modality]
    model = enc_cls(num_classes=n_classes, pretrained=args.pretrained).to(device)

    train_loader, val_loader, classes, counts = build_loaders(
        args.data_dir, model, args.batch_size, args.val_frac,
        args.seed, args.num_workers)
    print(f"Modality={args.modality}  classes={classes}  "
          f"train_counts={counts}")

    weights = class_weights_from_counts(counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    epochs = 1 if args.smoke_test else args.epochs
    best_path = f"{args.out_dir}/{args.modality}_best.pt"

    # Resume from checkpoint if requested
    if args.resume and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        stopper.best = ckpt["val_metrics"]["macro_f1"]
        print(f"Resumed from {best_path} (best macroF1={stopper.best:.3f})")

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"[{epoch:03d}/{epochs}] "
              f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.3f} | "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.3f} "
              f"macroF1 {va['macro_f1']:.3f}")

        is_best = stopper.step(va["macro_f1"])
        if is_best:
            save_checkpoint({
                "modality": args.modality,
                "state_dict": model.state_dict(),
                "classes": classes,
                "val_metrics": va,
                "epoch": epoch,
            }, best_path)
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} "
                  f"(best macroF1={stopper.best:.3f})")
            break

    print(f"Done. Best checkpoint -> {best_path}")


if __name__ == "__main__":
    main()
