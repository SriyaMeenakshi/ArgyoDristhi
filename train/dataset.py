"""
ArogyaDrishti — Dataset loader
Reads a CSV with columns: image_path, label
Resizes all images to 256x256 and applies augmentation during training.
"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_transforms(image_size=256, is_train=True):
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class ArogyaDataset(Dataset):
    def __init__(self, csv_path, image_size=256, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.transform = get_transforms(image_size, is_train)

        # Validate columns
        assert 'image_path' in self.df.columns, "CSV must have 'image_path' column"
        assert 'label' in self.df.columns, "CSV must have 'label' column"

        # Drop rows with missing files
        valid = self.df['image_path'].apply(os.path.exists)
        dropped = (~valid).sum()
        if dropped > 0:
            print(f"Warning: dropping {dropped} rows with missing image files")
        self.df = self.df[valid].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(Image.open(row['image_path']).convert('RGB'))
        augmented = self.transform(image=image)
        return augmented['image'], torch.tensor(row['label'], dtype=torch.long)
