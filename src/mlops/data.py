from pathlib import Path

import pandas as pd
import torch
import typer
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    """
    Custom dataset for GTSRB.

    This class loads preprocessed data saved as .pt files.
    """

    def __init__(self, data_path: Path, train: bool = True, transform=None) -> None:
        """
        Args:
            data_path (Path): Path to the directory containing processed .pt files.
            train (bool): Whether to load the training or test set.
            transform (callable, optional): Optional transform to be applied on a sample for data augmentation.
        """
        self.transform = transform
        split = "train" if train else "test"
        processed_path = Path(data_path) / f"{split}.pt"

        if not processed_path.exists():
            raise FileNotFoundError(
                f"{processed_path} not found. " "Please run `invoke preprocess-data` first."
            )

        self.images, self.labels = torch.load(processed_path)

        if self.transform:
            self.pil_transform = transforms.ToPILImage()
        else:
            self.pil_transform = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            # Convert tensor to PIL image to apply transformations
            image = self.pil_transform(image)
            image = self.transform(image)

        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """
        This method is a placeholder. The main preprocessing logic is in the
        `preprocess` function below.
        """
        raise NotImplementedError("Use the `preprocess` function at the module level.")


def preprocess(data_path: Path, output_folder: Path) -> None:
    """
    Preprocess the raw GTSRB data from `data_path` and save it to `output_folder`.
    It creates `train.pt` and `test.pt` files containing tensors of images and labels.
    """
    print(f"Preprocessing data from {data_path} and saving to {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    preprocess_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # Process training data
    print("Processing training data...")
    train_images, train_labels = [], []
    train_folder = data_path / "Train"
    if not train_folder.exists():
        raise FileNotFoundError(f"Training data not found at {train_folder}. Check your data directory.")
    for class_dir in sorted(train_folder.iterdir()):
        if class_dir.is_dir():
            class_id = int(class_dir.name)
            for img_path in class_dir.glob("*.png"):
                image = Image.open(img_path).convert("RGB")
                train_images.append(preprocess_transform(image))
                train_labels.append(class_id)

    torch.save((torch.stack(train_images), torch.tensor(train_labels, dtype=torch.long)), output_folder / "train.pt")
    print(f"Saved train.pt with {len(train_labels)} images to {output_folder}")

    # Process test data
    print("Processing test data...")
    test_images, test_labels = [], []
    test_csv = data_path / "Test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test metadata not found at {test_csv}. Check your data directory.")
    df = pd.read_csv(test_csv)
    for _, row in df.iterrows():
        img_path = data_path / row["Path"]
        image = Image.open(img_path).convert("RGB")
        test_images.append(preprocess_transform(image))
        test_labels.append(row["ClassId"])

    torch.save((torch.stack(test_images), torch.tensor(test_labels, dtype=torch.long)), output_folder / "test.pt")
    print(f"Saved test.pt with {len(test_labels)} images to {output_folder}")


if __name__ == "__main__":
    typer.run(preprocess)
