import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

class GTSRB(Dataset):
    def __init__(self, root_dir: str, mode: str = "train", transform= None, split = (0.8, 0.2)):
        """
        Args:
            root_dir (str or Path): Path to the directory containing 'Train', 'Test', and 'Test.csv'.
            train (bool): If True, loads the training set (folder structure). 
                          If False, loads the test set (CSV based).
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.samples = []
        self.split = split

        print("aaaa")
        if self.mode == "train":
            # --- Training Data Logic (Folder-based) ---
            # Looks for a folder named 'Train' (or 'train')
            # Inside are folders 0, 1, 2... representing classes
            self.data_folder = self.root_dir / "Train"
            df = pd.read_csv(self.root_dir / "Train.csv")
            df["track_id"] = df["Path"].str.split("/").str[-1].str.split("_").str[1]
          

            # 1. Initialize the splitter
            # n_splits=1 means just one train/val pair. test_size=0.2 means 20% val.
            #! random state hydra
            gss = GroupShuffleSplit(n_splits=1, test_size=self.split[1], random_state=42)

            # 2. Split
            # Notice we pass 'groups=df['track_id']'
            train_idx, val_idx = next(gss.split(X=df, y=df['ClassId'], groups=df['track_id']))

            # 3. Create Dataframes
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            print(f"val shape {val_df.shape}")
            print(f"train shape {train_df.shape}")
            # for _, row in df.iterrows():
            #     img_path
            
            


            # Loop through class folders (0, 1, 2...)
            #! get label from csv, not dir name
            # for class_id in os.listdir(self.data_folder):
            #     class_dir = self.data_folder / class_id
            #     if class_dir.is_dir():
            #         try:
            #             label = int(class_id)
            #             # Add every image in this class folder to our list
            #             for img_name in os.listdir(class_dir):
            #                 if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            #                     self.samples.append((class_dir / img_name, label))
            #         except ValueError:
            #             continue # Skip non-integer folders if any

        # else:
        #     # --- Test Data Logic (CSV-based) ---
        #     # Looks for Test.csv in the root
        #     csv_path = self.root_dir / "Test.csv"
            
        #     # Read the CSV to get paths and labels
        #     df = pd.read_csv(csv_path)

        #     for _, row in df.iterrows():
        #         # CSV Path example: "Test/00000.png"
        #         img_path = self.root_dir / row['Path']
        #         label = int(row['ClassId'])
        #         self.samples.append((img_path, label))

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

if __name__ == "__main__":
    path_raw = "data/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/versions/1/"
    obj = GTSRB(path_raw, mode="train", split=(0.8, 0.2))