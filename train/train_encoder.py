"""
ArogyaDrishti — Generic encoder trainer
Usage:
  python train/train_encoder.py --encoder face --train data/splits/face_train.csv --val data/splits/face_val.csv
  python train/train_encoder.py --encoder tongue --train data/splits/tongue_train.csv --val data/splits/tongue_val.csv

All encoder types: face, eye, tongue, skin, nail
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.dataset import ArogyaDataset


def load_encoder(encoder_name, num_classes):
    if encoder_name == 'face':
        from encoders.face_encoder import FaceEncoder
        return FaceEncoder(num_classes=num_classes)
    elif encoder_name == 'tongue':
        from encoders.tongue_encoder import TongueEncoder
        return TongueEncoder(num_classes=num_classes)
    elif encoder_name == 'eye':
        from encoders.eye_encoder import EyeEncoder
        return EyeEncoder(num_classes=num_classes)
    elif encoder_name == 'skin':
        from encoders.skin_encoder import SkinEncoder
        return SkinEncoder(num_classes=num_classes)
    elif encoder_name == 'nail':
        from encoders.nail_encoder import NailEncoder
        return NailEncoder(num_classes=num_classes)
    elif encoder_name == 'palm':
        from encoders.palm_encoder import PalmEncoder
        return PalmEncoder(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for images, labels in tqdm(loader, desc='Train', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    for images, labels in tqdm(loader, desc='Val  ', leave=False):
        images, labels = images.to(device), labels.to(device)
        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    # AUC — only meaningful for binary or one-vs-rest multi-class
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr',
                                average='macro', labels=list(range(num_classes)))
    except Exception:
        # Fall back to accuracy as proxy when AUC can't be computed
        auc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, auc, confusion_matrix(all_labels, all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', required=True, choices=['face','eye','tongue','skin','nail','palm'])
    parser.add_argument('--train', required=True, help='Path to train CSV')
    parser.add_argument('--val', required=True, help='Path to val CSV')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--wandb_project', default='arogyadrishti')
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'checkpoints/{args.encoder}'
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Datasets
    train_ds = ArogyaDataset(args.train, args.image_size, is_train=True)
    val_ds = ArogyaDataset(args.val, args.image_size, is_train=False)
    # num_workers=0 on Windows avoids multiprocessing warning spam
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # Model
    model = load_encoder(args.encoder, args.num_classes).to(device)

    # Class-weighted loss to handle imbalance
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    # W&B
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=f'{args.encoder}_encoder',
                   config=vars(args))

    best_auc = 0.0
    print(f"\nStarting training: {args.encoder} encoder, {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                criterion, scaler, device)
        val_loss, val_acc, val_auc, cm = validate(model, val_loader, criterion,
                                                   device, args.num_classes)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} AUC: {val_auc:.4f}")
        print(f"Confusion matrix:\n{cm}\n")

        if not args.no_wandb:
            wandb.log({'epoch': epoch, 'train/loss': train_loss, 'train/acc': train_acc,
                       'val/loss': val_loss, 'val/acc': val_acc, 'val/auc': val_auc,
                       'lr': optimizer.param_groups[0]['lr']})

        # Save best model — use accuracy as fallback when AUC is nan (single class)
        import math
        score = val_acc if (math.isnan(val_auc) or val_auc == 0.0) else val_auc
        if score > best_auc:
            best_auc = score
            save_path = os.path.join(args.output_dir, f'{args.encoder}_best.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_acc': val_acc, 'val_auc': val_auc,
                        'args': vars(args)}, save_path)
            print(f"  => Saved best model (AUC {val_auc:.4f}) to {save_path}")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f'{args.encoder}_epoch{epoch}.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)

    if not args.no_wandb:
        wandb.finish()
    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
