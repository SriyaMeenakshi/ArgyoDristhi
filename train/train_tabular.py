"""
ArogyaDrishti — Tabular Encoder Training Script

Usage:
  python train/train_tabular.py --epochs 50 --batch_size 256 --no_wandb

Input:  data/splits/tabular_train.csv
        data/splits/tabular_val.csv
Output: checkpoints/tabular/tabular_best.pt
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from encoders.tabular_encoder import TabularEncoder, FEATURES, TARGETS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset ────────────────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        missing = [c for c in FEATURES + TARGETS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.X = torch.tensor(self.df[FEATURES].values, dtype=torch.float32)
        self.Y = torch.tensor(
            self.df[[t for t in TARGETS]].values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        optimizer.zero_grad()
        with autocast('cuda'):
            preds = model(X)                             # dict {name: (batch,)}
            pred_stack = torch.stack(
                [preds[k.replace('risk_', '')] for k in TARGETS], dim=1
            )
        loss = criterion(pred_stack.float(), Y.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            with autocast('cuda'):
                preds = model(X)
                pred_stack = torch.stack(
                    [preds[k.replace('risk_', '')] for k in TARGETS], dim=1
                )
            loss = criterion(pred_stack.float(), Y.float())
            total_loss += loss.item()
            all_preds.append(pred_stack.cpu().numpy())
            all_labels.append(Y.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Per-head AUC
    aucs = []
    for i, name in enumerate(TARGETS):
        if len(np.unique(all_labels[:, i])) > 1:
            aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
    mean_auc = np.mean(aucs) if aucs else 0.0
    return total_loss / len(loader), mean_auc, aucs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',       default='data/splits/tabular_train.csv')
    parser.add_argument('--val',         default='data/splits/tabular_val.csv')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--no_wandb',    action='store_true')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="arogyadrishti", name="tabular_encoder",
                       config=vars(args))
        except Exception:
            use_wandb = False

    # Data
    train_ds = TabularDataset(args.train)
    val_ds   = TabularDataset(args.val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds):,} rows | Val: {len(val_ds):,} rows")

    # Model
    model = TabularEncoder(dropout=args.dropout).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    ckpt_dir = Path("checkpoints/tabular")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    print(f"\nStarting tabular encoder training for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        val_loss, mean_auc, per_head_aucs = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | "
              f"Mean AUC: {mean_auc:.4f}")

        for name, auc in zip(TARGETS, per_head_aucs):
            print(f"  {name:30s}: {auc:.4f}")

        if use_wandb:
            import wandb
            log = {'train_loss': train_loss, 'val_loss': val_loss,
                   'mean_auc': mean_auc}
            for name, auc in zip(TARGETS, per_head_aucs):
                log[f"auc_{name}"] = auc
            wandb.log(log, step=epoch)

        # Save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       ckpt_dir / f"tabular_epoch{epoch}.pt")

        # Save best
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), ckpt_dir / "tabular_best.pt")
            print(f"  => Saved best model (AUC {best_auc:.4f})")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Checkpoint: checkpoints/tabular/tabular_best.pt")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
