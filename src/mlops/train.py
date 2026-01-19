import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import os

# Assuming your GTSRB class is in a file named `dataset.py`
from .data.dataset import GTSRB
from .model import TinyCNN

# Initialize logger
log = logging.getLogger(__name__)

@hydra.main(config_path="../../configshydra", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    #models dir
    model_dir = hydra.utils.to_absolute_path(cfg.paths.models)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
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
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Verification (Optional) ---
    log.info(f"Train Dataset size: {len(train_ds)}")
    log.info(f"Validation Dataset size: {len(val_ds)}")

    # Try fetching one batch to ensure paths and transforms work
    images, labels = next(iter(train_loader))
    log.info(f"Batch Image Shape: {images.shape}")  # Should be [32, 3, 32, 32]
    log.info(f"Batch Label Shape: {labels.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyCNN(num_classes=cfg.data.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)

    epochs = cfg.training.epochs
    best_val_loss = None
    for epoch in range(epochs):
        model.train()

        for img, labels in train_loader:
            img, labels = img.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for img, labels in val_loader:
                img, labels = img.to(device), labels.to(device) 
                outputs = model(img)
                pred = outputs.argmax(dim=1, keepdim=True)  # Get predicted class
                correct += pred.eq(labels.view_as(pred)).sum().item()
                val_loss += criterion(outputs,labels).item()
            val_loss /= len(val_loader)
            log.info('\nEpoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, loss.item(),val_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        if not best_val_loss:
            best_val_loss= val_loss
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            #save to outputs
            save_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(model, save_path)
            #save to models/output
            save_path = os.path.join(model_dir, output_dir, "best_model.pt")
            torch.save(model, save_path)
            log.info(f"Model saved to {save_path}")
            log.info(f"Model saved to {output_dir}")
        
if __name__ == "__main__":
    main()
