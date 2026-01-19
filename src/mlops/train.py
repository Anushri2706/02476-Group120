import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn

# Assuming your GTSRB class is in a file named `dataset.py`
from .data.dataset import GTSRB
from .model import TinyCNN


@hydra.main(config_path="../../configshydra", config_name="config")
def main(cfg: DictConfig):
    # 1. Define Transforms
    # We must resize images to the same size so they can be stacked into a batch tensor.
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    # 2. Initialize Datasets using paths from Hydra config
    train_ds = GTSRB(
        raw_dir=hydra.utils.to_absolute_path(cfg.data.raw_dir),
        processed_dir=hydra.utils.to_absolute_path(cfg.data.processed_dir),
        mode="train",
        transform=transform,
    )

    val_ds = GTSRB(
        raw_dir=hydra.utils.to_absolute_path(cfg.data.raw_dir),
        processed_dir=hydra.utils.to_absolute_path(cfg.data.processed_dir),
        mode="val",
        transform=transform,
    )

    test_ds = GTSRB(
        raw_dir=hydra.utils.to_absolute_path(cfg.data.raw_dir),
        processed_dir=hydra.utils.to_absolute_path(cfg.data.processed_dir),
        mode="test",
        transform=transform,
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # --- Verification (Optional) ---
    print(f"Train Dataset size: {len(train_ds)}")
    print(f"Validation Dataset size: {len(val_ds)}")

    # Try fetching one batch to ensure paths and transforms work
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  # Should be [32, 3, 32, 32]
    print(f"Batch Label Shape: {labels.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyCNN(num_classes=43)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 3
    for epoch in range(epochs):
        model.train()

        for img, labels in train_loader:
            img, labels = img.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
