from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from torch.utils.data import Dataset

from src.mlops.data.dataset import GTSRB


# 1. Setup: Create a fake data folder with predictable data
@pytest.fixture
def dummy_data_path(tmp_path: Path) -> tuple[Path, Path]:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    # Create a fake training CSV with 20 items and labels 0-9
    train_labels = list(range(10)) * 2
    pd.DataFrame({"Path": [f"img_{i}.png" for i in range(20)], "ClassId": train_labels}).to_csv(
        processed_dir / "train_split.csv", index=False
    )

    # Create a fake testing CSV with 5 items and labels 0-4
    pd.DataFrame({"Path": [f"img_{i}.png" for i in range(5)], "ClassId": range(5)}).to_csv(
        processed_dir / "test.csv", index=False
    )

    # Create dummy image files (the test doesn't need to open them, but the dataset class expects them to exist)
    for i in range(20):
        Image.new("RGB", (32, 32)).save(raw_dir / f"img_{i}.png")

    return raw_dir, processed_dir


# 2. The Test: Check lengths, shapes, and label completeness
def test_dataset_properties(dummy_data_path: tuple[Path, Path]):
    raw_dir, processed_dir = dummy_data_path

    # --- Test Training Set ---
    train_dataset = GTSRB(raw_dir=raw_dir, processed_dir=processed_dir, mode="train")

    # Check length
    assert len(train_dataset) == 20

    # Check shape/type of a single data point
    img, label = train_dataset[0]
    assert isinstance(img, Image.Image)
    assert img.size == (32, 32)
    assert isinstance(label, int)

    # Check that all labels are represented
    train_labels_in_dataset = {sample[1] for sample in train_dataset.samples}
    assert train_labels_in_dataset == set(range(10))

    # --- Test Test Set ---
    test_dataset = GTSRB(raw_dir=raw_dir, processed_dir=processed_dir, mode="test")
    assert len(test_dataset) == 5
    test_labels_in_dataset = {sample[1] for sample in test_dataset.samples}
    assert test_labels_in_dataset == set(range(5))


def test_file_not_found(dummy_data_path: tuple[Path, Path]):
    """Tests that FileNotFoundError is raised for a missing CSV."""
    raw_dir, processed_dir = dummy_data_path
    with pytest.raises(FileNotFoundError):
        # The "val_split.csv" does not exist, so this should fail
        GTSRB(raw_dir=raw_dir, processed_dir=processed_dir, mode="val")
