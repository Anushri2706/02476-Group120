import torch
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.classification import MulticlassConfusionMatrix

from data.dataset import GTSRB
from model import TinyCNN

@hydra.main(config_path="../../configshydra", config_name="config", version_base="1.2")
def visualize(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Checkpoint
    ckpt_path = hydra.utils.to_absolute_path(cfg.ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    saved_cfg = checkpoint['config']
    
    # 2. Init Model
    model = TinyCNN(num_classes=saved_cfg.data.num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    #! resize should be hydraconfigurable
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    test_ds = GTSRB(
        raw_dir=hydra.utils.to_absolute_path(cfg.data.raw_dir),
        processed_dir=hydra.utils.to_absolute_path(cfg.data.processed_dir),
        mode="test",
        transform=transform
    )
    # num_workers=0 to prevent hanging
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # 4. Inference
    preds, targets = [], []
    with torch.no_grad():
        for img, label in test_loader:
            #most likely class
            preds.append(torch.argmax(model(img.to(device)), dim=1).cpu())
            targets.append(label)
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    #standard sns implementation of the confusion matrix
    confmat = MulticlassConfusionMatrix(num_classes=saved_cfg.data.num_classes)(preds, targets)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confmat.float() / (confmat.sum(axis=1, keepdim=True) + 1e-9), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    save_dir = hydra.utils.to_absolute_path(cfg.figures_path)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    print(f"Saved figure to {save_dir}/confusion_matrix.png")

if __name__ == "__main__":
    visualize()

