import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

class GTSRB(Dataset):
    def __init__(self, raw_dir: str, processed_dir: str, mode: str = "train", transform= None):
        """
        Args:
            raw_dir (str or Path): Path to the raw images (data/raw/gtsrb)
            processed_dir (str or Path): Path to the CSV splits (data/processed)
            mode (str): 'train', 'val', or 'test'. Defaults to "train".
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        self.transform = transform
        self.samples = []
        self.mode = mode
        df = None
        if self.mode == "train":
            csv_path = self.processed_dir / "train_split.csv"
        elif self.mode == "val":
            csv_path = self.processed_dir / "val_split.csv"
        elif self.mode == "test":
            csv_path = self.processed_dir / "test.csv"

        df = pd.read_csv(csv_path)
        self.samples = [
            (self.raw_dir / p, label) 
            for p, label in zip(df['Path'], df['ClassId'])
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # open image and convert to RGB (standard for PyTorch)
            image = Image.open(img_path).convert("RGB")

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e

        if self.transform:
            image = self.transform(image)
        return image, label
