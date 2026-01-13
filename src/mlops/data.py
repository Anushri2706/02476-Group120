import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from pathlib import Path
from PIL import Image

class GTSRB(Dataset):
    def __init__(self, root_dir: str, train: bool = True, transform=None):
        """
        Args:
            root_dir (str or Path): Path to the directory containing 'Train', 'Test', and 'Test.csv'.
            train (bool): If True, loads the training set (folder structure). 
                          If False, loads the test set (CSV based).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train
        self.samples = []

        if self.train:
            # --- Training Data Logic (Folder-based) ---
            # Looks for a folder named 'Train' (or 'train')
            # Inside are folders 0, 1, 2... representing classes
            self.data_folder = self.root_dir / "Train"
            if not self.data_folder.exists():
                self.data_folder = self.root_dir / "train"
            
            if not self.data_folder.exists():
                 raise FileNotFoundError(f"Could not find 'Train' or 'train' folder in {self.root_dir}")

            # Loop through class folders (0, 1, 2...)
            for class_id in os.listdir(self.data_folder):
                class_dir = self.data_folder / class_id
                if class_dir.is_dir():
                    try:
                        label = int(class_id)
                        # Add every image in this class folder to our list
                        for img_name in os.listdir(class_dir):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((class_dir / img_name, label))
                    except ValueError:
                        continue # Skip non-integer folders if any

        else:
            # --- Test Data Logic (CSV-based) ---
            # Looks for Test.csv in the root
            csv_path = self.root_dir / "Test.csv"
            if not csv_path.exists():
                 raise FileNotFoundError(f"Could not find 'Test.csv' in {self.root_dir}")
            
            # Read the CSV to get paths and labels
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                # CSV Path example: "Test/00000.png"
                img_path = self.root_dir / row['Path']
                label = int(row['ClassId'])
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # open image and convert to RGB (standard for PyTorch)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor or handle error as preferred
            # For now, we crash to let you know something is wrong
            raise e

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Sanity Check Block ---
if __name__ == "__main__":
    # This block only runs if you execute "python src/mlops/data.py"
    # It allows you to test this specific file in isolation.
    
    # Define the path relative to where you run the command (project root)
    # Update this path if yours is different!
    TEST_PATH = Path("data/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/versions/1")
    
    if TEST_PATH.exists():
        print(f"Testing GTSRB Dataset from: {TEST_PATH}")
        
        # Test 1: Training Set
        print("\n--- Checking Training Set ---")
        try:
            train_ds = GTSRB(TEST_PATH, train=True)
            print(f"✅ Successfully loaded Training Set.")
            print(f"   Total images: {len(train_ds)}")
            if len(train_ds) > 0:
                img, label = train_ds[0]
                print(f"   Sample 0 - Label: {label}, Image Size: {img.size}")
        except Exception as e:
             print(f"❌ Failed to load Training Set: {e}")

        # Test 2: Test Set
        print("\n--- Checking Test Set ---")
        try:
            test_ds = GTSRB(TEST_PATH, train=False)
            print(f"✅ Successfully loaded Test Set.")
            print(f"   Total images: {len(test_ds)}")
        except Exception as e:
             print(f"❌ Failed to load Test Set: {e}")
    else:
        print(f"❌ Path not found: {TEST_PATH}")
        print("Run this script from the project root or fix the TEST_PATH variable.")