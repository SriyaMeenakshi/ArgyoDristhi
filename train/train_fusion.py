"""
ArogyaDrishti — Fusion Module Training Script

Loads all trained encoder checkpoints, extracts embeddings,
then trains the cross-modal attention fusion module with 7 risk heads.

Since we don't have paired multi-modal patient data, this script:
  1. Extracts embeddings from each encoder's validation set
  2. Creates synthetic multi-modal batches by randomly combining embeddings
  3. Uses tabular risk labels as supervision targets
  4. Trains only the FusionModule (encoders are frozen)

Usage:
  python train/train_fusion.py --no_wandb
  python train/train_fusion.py --epochs 30 --batch_size 32 --no_wandb
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoders.face_encoder    import FaceEncoder
from encoders.eye_encoder     import EyeEncoder
from encoders.tongue_encoder  import TongueEncoder
from encoders.skin_encoder    import SkinEncoder
from encoders.nail_encoder    import NailEncoder
from encoders.palm_encoder    import PalmEncoder
from encoders.tabular_encoder import TabularEncoder, FEATURES, TARGETS
from encoders.fusion_module   import FusionModule
from train.dataset            import ArogyaDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Encoder config ─────────────────────────────────────────────────────────────
ENCODER_CONFIG = {
    'face':    {'cls': FaceEncoder,    'num_classes': 4,  'embed_dim': 768,  'ckpt': 'checkpoints/face/face_best.pt'},
    'eye':     {'cls': EyeEncoder,     'num_classes': 2,  'embed_dim': 768,  'ckpt': 'checkpoints/eye/eye_best.pt'},
    'tongue':  {'cls': TongueEncoder,  'num_classes': 4,  'embed_dim': 1024, 'ckpt': 'checkpoints/tongue/tongue_best.pt'},
    'skin':    {'cls': SkinEncoder,    'num_classes': 7,  'embed_dim': 768,  'ckpt': 'checkpoints/skin/skin_best.pt'},
    'nail':    {'cls': NailEncoder,    'num_classes': 6,  'embed_dim': 1792, 'ckpt': 'checkpoints/nail/nail_best.pt'},
    'palm':    {'cls': PalmEncoder,    'num_classes': 5,  'embed_dim': 1536, 'ckpt': 'checkpoints/palm/palm_best.pt'},
    'tabular': {'cls': TabularEncoder, 'num_classes': 7,  'embed_dim': 100,  'ckpt': 'checkpoints/tabular/tabular_best.pt'},
}

IMAGE_SPLITS = {
    'face':   ('data/splits/face_val.csv',    224),
    'eye':    ('data/splits/eye_val.csv',     224),
    'tongue': ('data/splits/tongue_val.csv',  224),
    'skin':   ('data/splits/skin_val.csv',    224),
    'nail':   ('data/splits/nail_val.csv',    224),
    'palm':   ('data/splits/palm_val.csv',    224),
}


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: Load encoder + extract embeddings
# ══════════════════════════════════════════════════════════════════════════════

def load_encoder(name: str, cfg: dict):
    """Load encoder and its checkpoint."""
    print(f"  Loading {name} encoder...")
    if name == 'tabular':
        model = cfg['cls']().to(DEVICE)
    else:
        model = cfg['cls'](num_classes=cfg['num_classes']).to(DEVICE)

    ckpt_path = cfg['ckpt']
    if Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print(f"    Loaded checkpoint: {ckpt_path}")
    else:
        print(f"    WARNING: No checkpoint at {ckpt_path} — using random weights")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_image_embeddings(encoder, csv_path: str, image_size: int,
                              batch_size: int = 32) -> np.ndarray:
    """Extract embeddings from image encoder."""
    if not Path(csv_path).exists():
        return None
    ds = ArogyaDataset(csv_path, image_size=image_size, is_train=False)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs = []
    for images, _ in loader:
        images = images.to(DEVICE)
        with autocast('cuda'):
            emb = encoder.get_embedding(images)
        embs.append(emb.cpu().float().numpy())
    return np.vstack(embs)


@torch.no_grad()
def extract_tabular_embeddings(encoder, csv_path: str,
                                batch_size: int = 256) -> tuple:
    """Extract tabular embeddings + risk labels."""
    if not Path(csv_path).exists():
        return None, None
    df = pd.read_csv(csv_path)
    X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    Y = torch.tensor(df[[t for t in TARGETS]].values, dtype=torch.float32)
    loader = DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        batch_size=batch_size, shuffle=False, num_workers=0)
    embs, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        with autocast('cuda'):
            emb = encoder.get_embedding(x)
        embs.append(emb.cpu().float().numpy())
        labels.append(y.numpy())
    return np.vstack(embs), np.vstack(labels)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: Synthetic multi-modal dataset
# ══════════════════════════════════════════════════════════════════════════════

class FusionDataset(Dataset):
    """
    Creates synthetic multi-modal batches by randomly sampling one embedding
    from each encoder. Labels come from tabular risk scores.
    """
    def __init__(self, embeddings: dict, labels: np.ndarray, n_samples: int = 10000):
        self.embeddings = {k: torch.tensor(v, dtype=torch.float32)
                           for k, v in embeddings.items()}
        self.labels  = torch.tensor(labels, dtype=torch.float32)
        self.n       = n_samples
        self.n_label = len(labels)

        # Size of each encoder's pool
        self.sizes = {k: len(v) for k, v in embeddings.items()}

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Anchor: tabular embedding + its real paired label
        tab_idx = idx % self.n_label
        sample = {'tabular': self.embeddings['tabular'][tab_idx]}
        label  = self.labels[tab_idx]
        # Context: randomly sample image embeddings per encoder
        for name, embs in self.embeddings.items():
            if name == 'tabular':
                continue
            i = np.random.randint(0, self.sizes[name])
            sample[name] = embs[i]
        return sample, label


def collate_fusion(batch):
    samples, labels = zip(*batch)
    keys = samples[0].keys()
    collated = {k: torch.stack([s[k] for s in samples]) for k in keys}
    return collated, torch.stack(labels)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: Train / evaluate fusion module
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(fusion, loader, optimizer, scaler, criterion):
    fusion.train()
    total_loss = 0.0
    for embeddings, labels in loader:
        embeddings = {k: v.to(DEVICE) for k, v in embeddings.items()}
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        risks = fusion(embeddings)
        pred = torch.stack([v for v in risks.values() if isinstance(v, torch.Tensor)], dim=1)
        pred = pred.float().clamp(1e-7, 1 - 1e-7)
        loss = criterion(pred, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(fusion, loader, criterion):
    fusion.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = {k: v.to(DEVICE) for k, v in embeddings.items()}
            labels = labels.to(DEVICE)
            risks = fusion(embeddings)
            pred = torch.stack([v for v in risks.values() if isinstance(v, torch.Tensor)], dim=1)
            pred = pred.float().clamp(1e-7, 1 - 1e-7)
            loss = criterion(pred, labels.float())
            total_loss += loss.item()
            all_preds.append(pred.cpu().float().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    from sklearn.metrics import roc_auc_score
    aucs = []
    bin_labels = (all_labels >= 0.5).astype(int)
    for i in range(all_labels.shape[1]):
        if len(np.unique(bin_labels[:, i])) > 1:
            aucs.append(roc_auc_score(bin_labels[:, i], all_preds[:, i]))
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    return total_loss / len(loader), mean_auc


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--n_train',    type=int,   default=10000)
    parser.add_argument('--n_val',      type=int,   default=2000)
    parser.add_argument('--no_wandb',   action='store_true')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print("\n[1] Loading encoders and extracting embeddings...")

    # Load all encoders
    encoders = {}
    for name, cfg in ENCODER_CONFIG.items():
        encoders[name] = load_encoder(name, cfg)

    # Extract embeddings
    all_embs = {}
    for name in ['face', 'eye', 'tongue', 'skin', 'nail', 'palm']:
        csv, img_size = IMAGE_SPLITS[name]
        embs = extract_image_embeddings(encoders[name], csv, img_size)
        if embs is not None:
            all_embs[name] = embs
            print(f"  {name}: {embs.shape[0]} embeddings, dim={embs.shape[1]}")
        else:
            # Use random embeddings as fallback
            dim = ENCODER_CONFIG[name]['embed_dim']
            all_embs[name] = np.random.randn(500, dim).astype(np.float32)
            print(f"  {name}: No data — using {500} random embeddings (dim={dim})")

    # Tabular embeddings + labels
    tab_embs, tab_labels = extract_tabular_embeddings(
        encoders['tabular'], 'data/splits/tabular_val.csv')
    if tab_embs is not None:
        all_embs['tabular'] = tab_embs
        print(f"  tabular: {tab_embs.shape[0]} embeddings, dim={tab_embs.shape[1]}")
    else:
        all_embs['tabular'] = np.random.randn(500, 100).astype(np.float32)
        tab_labels = np.random.randint(0, 2, (500, 7)).astype(np.float32)
        print("  tabular: No data — using random embeddings")

    
    # Build fusion module
    encoder_dims = {name: all_embs[name].shape[1] for name in all_embs}
    fusion = FusionModule(encoder_dims=encoder_dims).to(DEVICE)


    # Blend tabular labels (60%) with encoder-predicted labels (40%)
    encoder_labels = extract_encoder_risk_labels(encoders, all_embs)
    n_tab = len(tab_labels)
    encoder_labels_trimmed = encoder_labels[:n_tab]
    tab_labels = 0.6 * tab_labels + 0.4 * encoder_labels_trimmed
    print("  Labels blended: 60% tabular + 40% encoder predictions")

    print(f"\n[2] Building fusion datasets...")
    train_ds = FusionDataset(all_embs, tab_labels, n_samples=args.n_train)
    val_ds   = FusionDataset(all_embs, tab_labels, n_samples=args.n_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0,
                              collate_fn=collate_fusion)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0,
                              collate_fn=collate_fusion)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler('cuda')

    ckpt_dir = Path("checkpoints/fusion")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="arogyadrishti", name="fusion_module", config=vars(args))
        except Exception:
            use_wandb = False

    print(f"\n[3] Training fusion module for {args.epochs} epochs...")
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(fusion, train_loader, optimizer, scaler, criterion)
        val_loss, mean_auc = eval_epoch(fusion, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | "
              f"Mean AUC: {mean_auc:.4f}")

        if use_wandb:
            import wandb
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss,
                       'mean_auc': mean_auc}, step=epoch)

        if epoch % 10 == 0:
            torch.save(fusion.state_dict(), ckpt_dir / f"fusion_epoch{epoch}.pt")

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(fusion.state_dict(), ckpt_dir / "fusion_best.pt")
            print(f"  => Saved best fusion model (AUC {best_auc:.4f})")

    print(f"\nFusion training complete. Best AUC: {best_auc:.4f}")
    print(f"Checkpoint: checkpoints/fusion/fusion_best.pt")
    print("\nNext step: python export/export_tflite.py")

    if use_wandb:
        import wandb
        wandb.finish()




@torch.no_grad()
def extract_encoder_risk_labels(encoders: dict, all_embs: dict) -> np.ndarray:
    """
    Derive risk labels from encoder classifier predictions.
    Maps each encoder output class -> risk head probability.
    This makes fusion learn from actual image content, not just tabular.
    """
    n = max(len(v) for v in all_embs.values())
    labels = np.zeros((n, 7), dtype=np.float32)  # 7 risk heads

    # Nail: clubbing->cardiovascular, cyanosis->cardiovascular, 
    #        pitting->dermatological, ALM->dermatological
    if 'nail' in all_embs:
        nail_embs = torch.tensor(all_embs['nail']).to(DEVICE)
        nail_enc = encoders['nail']
        for i in range(0, len(nail_embs), 32):
            batch = nail_embs[i:i+32]
            logits = nail_enc.classifier(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            # class 1=clubbing, 2=cyanosis -> cardiovascular
            labels[i:i+len(probs), 4] = probs[:, 1] + probs[:, 2]
            # class 4=pitting, 5=ALM -> dermatological  
            labels[i:i+len(probs), 5] = probs[:, 4] + probs[:, 5]
            # class 0=healthy -> reduce all risks
            labels[i:i+len(probs), 0] *= (1 - probs[:, 0] * 0.5)

    # Face: pallor->hematological, genetic->nutritional
    if 'face' in all_embs:
        face_embs = torch.tensor(all_embs['face']).to(DEVICE)
        face_enc = encoders['face']
        idx = min(len(face_embs), n)
        for i in range(0, idx, 32):
            batch = face_embs[i:i+32]
            logits = face_enc.classifier(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            # class 1=pallor -> hematological
            labels[i:i+len(probs), 0] = np.maximum(
                labels[i:i+len(probs), 0], probs[:, 1])
            # class 3=genetic -> nutritional
            labels[i:i+len(probs), 6] = np.maximum(
                labels[i:i+len(probs), 6], probs[:, 3])

    # Palm: pallor->hematological, erythema->hepatic, jaundice->hepatic
    if 'palm' in all_embs:
        palm_embs = torch.tensor(all_embs['palm']).to(DEVICE)
        palm_enc = encoders['palm']
        idx = min(len(palm_embs), n)
        for i in range(0, idx, 32):
            batch = palm_embs[i:i+32]
            logits = palm_enc.classifier(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            # class 1=erythema, 3=jaundice -> hepatic
            labels[i:i+len(probs), 3] = np.maximum(
                labels[i:i+len(probs), 3], probs[:, 1] + probs[:, 3])
            # class 2=pallor -> hematological
            labels[i:i+len(probs), 0] = np.maximum(
                labels[i:i+len(probs), 0], probs[:, 2])

    return np.clip(labels, 0, 1)
    return np.clip(labels, 0, 1)

if __name__ == '__main__':
    main()

