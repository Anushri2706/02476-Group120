import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import os
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import MeanMetric


# Assuming your GTSRB class is in a file named `dataset.py`
from mlops.data.dataset import GTSRB
from mlops.model import TinyCNN

# Initialize logger
log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

@hydra.main(config_path="../../configshydra", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    #models dir
    model_dir = hydra.utils.to_absolute_path(cfg.paths.models)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # 1. Define Transforms
    # We must resize images to the same size so they can be stacked into a batch tensor.
    #make image size configurable
    image_size = tuple(cfg.image_size)
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

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

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)


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
    
    log.info(f"Starting training for {epochs} epochs on device: {device}")
    log.info(f"Model: {model.__class__.__name__}")
    log.info(f"Optimizer: {optimizer.__class__.__name__} (lr={cfg.training.lr})")
    log.info(f"Batch size: {cfg.training.batch_size}")

    num_classes = cfg.data.num_classes
    train_acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    val_acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    train_loss_metric = MeanMetric().to(device)
    val_loss_metric = MeanMetric().to(device)

    for epoch in range(epochs):
        model.train()
        log.info(f"Starting Epoch {epoch + 1}/{epochs}")
        
        total_batches = len(train_loader)
        for batch_idx, (img, labels) in enumerate(train_loader, 1):
            img, labels = img.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_metric.update(loss)
            train_acc_metric.update(outputs, labels)
            
            # Log progress every 10% of batches or every 50 batches
            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx % 50 == 0:
                log.info(f"  Epoch {epoch + 1}/{epochs} - Batch {batch_idx}/{total_batches} "
                        f"({100 * batch_idx / total_batches:.1f}%) - "
                        f"Loss: {loss.item():.4f}")

        log.info(f"Starting validation for Epoch {epoch + 1}/{epochs}")
        model.eval()

        with torch.no_grad():
            for img, labels in val_loader:
                img, labels = img.to(device), labels.to(device) 
                outputs = model(img)
                val_loss_metric.update(criterion(outputs, labels))
                val_acc_metric.update(outputs, labels)

        epoch_train_loss = train_loss_metric.compute()
        epoch_train_acc = train_acc_metric.compute()
        epoch_val_loss = val_loss_metric.compute()
        epoch_val_acc = val_acc_metric.compute()

        log.info(f"=" * 80)
        log.info(f'Epoch {epoch + 1}/{epochs} Complete:')
        log.info(f'  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2%}')
        log.info(f'  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2%}')
        log.info(f"=" * 80)

        # Reset metrics for the next epoch
        train_loss_metric.reset()
        train_acc_metric.reset()
        val_loss_metric.reset()
        val_acc_metric.reset()

        if best_val_loss is None or epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss
            log.info(f"✓ New best model saved! Val Loss: {best_val_loss:.4f}")
            #save with additional info because we may change the architecture of the model at 
            #some point and to open this this model we need to intialize the model as it was saved.
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'config': cfg, 
                'metric': best_val_loss
            }
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))
    
    log.info(f"\n{'='*80}")
    log.info(f"Training Complete!")
    log.info(f"Best Validation Loss: {best_val_loss:.4f}")
    log.info(f"{'='*80}\n")
    
    #saves after training to 'models/latest' helpful for now but we could eliminate it later
    best_model_path = os.path.join(output_dir, "best_model.pth")
    latest_dir = os.path.join(model_dir, "latest")
    os.makedirs(latest_dir, exist_ok=True)
    latest_save_path = os.path.join(latest_dir, "best_model.pth")
    # Load the checkpoint dict
    checkpoint = torch.load(best_model_path)
    # Save the checkpoint dict (preserves config)
    torch.save(checkpoint, latest_save_path)
        
if __name__ == "__main__":
    main()
