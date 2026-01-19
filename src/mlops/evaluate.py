import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

# Local imports
from .data.dataset import GTSRB
from .model import TinyCNN

def evaluate(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Checkpoint & Extract Config
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    cfg = checkpoint['config'] # Load config saved during training
    
    # 2. Initialize Model with Saved Params
    model = TinyCNN(num_classes=cfg.data.num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 3. Setup Data (using paths from the saved config)
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    test_ds = GTSRB(cfg.data.raw_dir, cfg.data.processed_dir, mode="test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    # 4. Metrics
    metrics = MetricCollection({
        'Accuracy': MulticlassAccuracy(num_classes=cfg.data.num_classes, average='micro'),
        'Precision': MulticlassPrecision(num_classes=cfg.data.num_classes, average='macro'),
        'Recall': MulticlassRecall(num_classes=cfg.data.num_classes, average='macro'),
        'F1': MulticlassF1Score(num_classes=cfg.data.num_classes, average='macro')
    }).to(device)

    # 5. Inference Loop
    with torch.no_grad():
        for img, labels in test_loader:
            img, labels = img.to(device), labels.to(device)
            outputs = model(img)
            metrics.update(outputs, labels)

    # 6. Output
    results = metrics.compute()
    print("-" * 30)
    print(f"Evaluation Results (Epoch {checkpoint.get('epoch', 'Unknown')})")
    print("-" * 30)
    for key, val in results.items():
        print(f"{key}: {val.item():.4f}")

if __name__ == "__main__":
    # TODO: Fill in your path here
    MODEL_PATH = "" 
    
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        evaluate(MODEL_PATH)
    else:
        print("Please provide a valid MODEL_PATH in the script.")