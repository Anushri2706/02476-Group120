import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import os
from hydra.utils import instantiate
from torchmetrics import MetricCollection, MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import wandb
from omegaconf import OmegaConf

# Assuming your GTSRB class is in a file named `dataset.py`
from .data.dataset import GTSRB
from .model import TinyCNN

# Initialize logger
log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

@hydra.main(config_path="../../configshydra", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    #models dir
    wandb.init(
        project=cfg.wandb.project_name, 
        entity=cfg.wandb.team_name,
        # Convert Hydra config to a standard Python dict for W&B
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        job_type="train" 
    )
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
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, pin_memory=True)


    # --- Verification (Optional) ---
    log.info(f"Train Dataset size: {len(train_ds)}")
    log.info(f"Validation Dataset size: {len(val_ds)}")

    # Try fetching one batch to ensure paths and transforms work
    images, labels = next(iter(train_loader))
    log.info(f"Batch Image Shape: {images.shape}")  # Should be [32, 3, 32, 32]
    log.info(f"Batch Label Shape: {labels.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #model = TinyCNN(num_classes=cfg.data.num_classes).to(device)
    model = instantiate(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    epochs = cfg.training.epochs
    best_val_loss = None
    
    log.info(f"Starting training for {epochs} epochs on device: {device}")
    log.info("Training configuration:\n" + OmegaConf.to_yaml(cfg.training, resolve=True))
    log.info("Model configuration:\n" + OmegaConf.to_yaml(cfg.model, resolve=True))


    num_classes = cfg.data.num_classes
    metrics_template = MetricCollection({
        'Accuracy': MulticlassAccuracy(num_classes=cfg.data.num_classes, average='micro'),
        'Precision': MulticlassPrecision(num_classes=cfg.data.num_classes, average='macro'),
        'Recall': MulticlassRecall(num_classes=cfg.data.num_classes, average='macro'),
        'F1': MulticlassF1Score(num_classes=cfg.data.num_classes, average='macro')
    })
    #clone the template to create to separate buckets for train and val so that they dont contaminate eachother
    train_metrics = metrics_template.clone(prefix='train_').to(device)
    val_metrics = metrics_template.clone(prefix='val_').to(device)
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
            train_metrics.update(outputs, labels)

        model.eval()

        with torch.no_grad():
            for img, labels in val_loader:
                img, labels = img.to(device), labels.to(device) 
                outputs = model(img)
                val_loss = criterion(outputs, labels)
                val_loss_metric.update(val_loss)
                val_metrics.update(outputs, labels)

        tr_metrics = train_metrics.compute()
        vl_metrics = val_metrics.compute()
        tr_loss = train_loss_metric.compute()
        vl_loss = val_loss_metric.compute()

        log_dict = {
            "epoch": epoch,
            "train_loss": tr_loss.item(),
            "val_loss": vl_loss.item(),
        }
        # Add all metric collection results to the dict
        # The keys already have 'train_' and 'val_' prefixes!
        log_dict.update({k: v.item() for k, v in tr_metrics.items()})
        log_dict.update({k: v.item() for k, v in vl_metrics.items()})

        # Log everything at once
        wandb.log(log_dict)

        log.info(f"Epoch {epoch} | Train Loss: {tr_loss:.4f} Acc: {tr_metrics['train_Accuracy']:.2%} | Val Loss: {vl_loss:.4f} Acc: {vl_metrics['val_Accuracy']:.2%}")

        # Reset metrics for next epoch
        train_metrics.reset()
        val_metrics.reset()
        train_loss_metric.reset()
        val_loss_metric.reset()

        if best_val_loss is None or vl_loss <= best_val_loss:
            best_val_loss = vl_loss
            #save with additional info because we may change the architecture of the model at 
            #some point and to open this this model we need to intialize the model as it was saved.
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'config': cfg, 
                'metric': best_val_loss
            }
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))
    wandb.finish()
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
