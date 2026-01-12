import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

def process_split(csv_path: Path, root_dir: Path, transform: transforms.Compose) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads a GTSRB CSV file, crops/resizes images, and returns tensors.
    
    Args:
        csv_path: Path to the CSV file (Train.csv or Test.csv).
        root_dir: Root directory containing image folders.
        transform: Torchvision transforms to apply.

    Returns:
        A tuple containing (data_tensor, labels_tensor).
    """
    df = pd.read_csv(csv_path)
    data = []
    labels = []

    # Total rows for progress tracking
    total = len(df)
    print(f"Processing {csv_path.name}: {total} images found.")

    for i, row in df.iterrows():
        # Path manipulation using pathlib
        img_path = root_dir / row['Path']
        
        try:
            with Image.open(img_path) as img:
                # Crop using ROI coordinates from CSV
                # Box format: (left, upper, right, lower)
                box = (row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2'])
                crop = img.crop(box)
                
                # Apply transformations (Resize -> ToTensor)
                tensor = transform(crop)
                
                data.append(tensor)
                labels.append(int(row['ClassId']))
        
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not process image {img_path}. Error: {e}")

        # Simple progress logger
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{total} images...")

    # Stack list of tensors into a single tensor
    return torch.stack(data), torch.tensor(labels)

def preprocess_gtsrb(raw_dir: str, processed_dir: str) -> None:
    """
    Main preprocessing pipeline for GTSRB.
    Checks for Train.csv and Test.csv, processes them, and saves .pt files.
    """
    raw_path = Path(raw_dir)
    save_path = Path(processed_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Define standard transformations
    # GTSRB images vary in size; we resize to 32x32 for standard CNNs
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # --- Process Test Data ---
    test_csv = raw_path / "Test.csv"
    if test_csv.exists():
        test_data, test_labels = process_split(test_csv, raw_path, transform)
        torch.save(test_data, save_path / "test_images.pt")
        torch.save(test_labels, save_path / "test_targets.pt")
        print(f"Saved Test data: {test_data.shape}")
    else:
        print(f"Warning: {test_csv} not found. Skipping Test set.")

    # --- Process Train Data ---
    train_csv = raw_path / "Train.csv"
    if train_csv.exists():
        train_data, train_labels = process_split(train_csv, raw_path, transform)
        torch.save(train_data, save_path / "train_images.pt")
        torch.save(train_labels, save_path / "train_targets.pt")
        print(f"Saved Train data: {train_data.shape}")
    else:
        print(f"Warning: {train_csv} not found. Skipping Train set.")


class GTSRBDataset(Dataset):
    """
    A custom dataset that wraps the pre-processed tensors 
    and applies on-the-fly augmentation.
    """
    def __init__(self, images_path, targets_path, transform=None):
        self.images = torch.load(images_path)
        self.targets = torch.load(targets_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.targets[idx]
        
        # Apply augmentations (e.g., RandomRotation) if provided
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_loaders(data_dir="data/processed", batch_size=64):
    """
    Returns train and test dataloaders.
    """
    data_path = Path(data_dir)
    
    # 1. Define Augmentations (Train only)
    # We normalized in preprocessing, but usually we need to re-verify 
    # if ToTensor() was already applied. 
    # Since our tensors are already floats 0-1, we just apply geometric changes.
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        # Add more here if needed
    ])

    # 2. Create Datasets
    train_set = GTSRBDataset(
        data_path / "train_images.pt", 
        data_path / "train_targets.pt", 
        transform=train_transform
    )
    
    test_set = GTSRBDataset(
        data_path / "test_images.pt", 
        data_path / "test_targets.pt", 
        transform=None # No augmentation for test!
    )

    # 3. Create Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Adjust these paths if your folder structure is different
    preprocess_gtsrb("data/raw", "data/processed")