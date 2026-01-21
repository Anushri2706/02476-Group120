from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from src.mlops.data.dataset import MyDataset


@pytest.fixture
def processed_data_path(tmp_path: Path) -> Path:
    """Create dummy processed data files for testing."""
    images = torch.randn(10, 3, 32, 32)
    labels = torch.randint(0, 43, (10,))
    torch.save((images, labels), tmp_path / "train.pt")
    torch.save((images, labels), tmp_path / "test.pt")
    return tmp_path


def test_my_dataset(processed_data_path: Path):
    """Test the MyDataset class."""
    # Test training dataset
    train_dataset = MyDataset(processed_data_path, train=True)
    assert isinstance(train_dataset, Dataset)
    assert len(train_dataset) == 10
    img, label = train_dataset[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, torch.Tensor)

    # Test test dataset
    test_dataset = MyDataset(processed_data_path, train=False)
    assert isinstance(test_dataset, Dataset)
    assert len(test_dataset) == 10


def test_my_dataset_file_not_found(tmp_path: Path):
    """Test that FileNotFoundError is raised for non-existent data."""
    with pytest.raises(FileNotFoundError):
        MyDataset(tmp_path)