import torch
import hydra
import os
import sys
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

from .data.dataset import GTSRB
from .model import TinyCNN
from hydra.utils import instantiate

@hydra.main(config_path="../../configshydra", config_name="config", version_base="1.2")
def evaluate(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = hydra.utils.to_absolute_path(cfg.ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    saved_cfg = checkpoint['config']
    model = instantiate(saved_cfg.model).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    image_size = tuple(saved_cfg.image_size)
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    test_ds = GTSRB(
        raw_dir=hydra.utils.to_absolute_path(cfg.data.raw_dir),
        processed_dir=hydra.utils.to_absolute_path(cfg.data.processed_dir),
        mode="test",
        transform=transform
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)


    metrics = MetricCollection({
        'Accuracy': MulticlassAccuracy(num_classes=saved_cfg.data.num_classes, average='micro'),
        'Precision': MulticlassPrecision(num_classes=saved_cfg.data.num_classes, average='macro'),
        'Recall': MulticlassRecall(num_classes=saved_cfg.data.num_classes, average='macro'),
        'F1': MulticlassF1Score(num_classes=saved_cfg.data.num_classes, average='macro')
    }).to(device)


    with torch.no_grad():
        for img, labels in test_loader:
            metrics.update(model(img.to(device)), labels.to(device))

    #just a print here no logging yet
    print(f"Results (Epoch {checkpoint['epoch']}):")
    for k, v in metrics.compute().items():
        print(f"{k}: {v.item():.4f}")

if __name__ == "__main__":
    evaluate()