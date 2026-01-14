from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from data.dataset import GTSRB

# 1. Define the Resize Logic Here
# This pipeline runs on every image right after it's loaded
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),       # <--- THIS IS THE RESIZE STEP
    transforms.ToTensor(),             # Converts 0-255 Image to 0-1 Tensor
])
current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent.parent 

# 2. Define Absolute Paths
raw_dir_path = project_root / "data" / "raw" / "gtsrb"
processed_dir_path = project_root / "data" / "processed"
#! hydra config this
train_dataset = GTSRB(
        raw_dir=raw_dir_path, 
        processed_dir=processed_dir_path, 
        mode="train", 
        transform=data_transform
    )

# 3. Test it
img, label = train_dataset[0]
print(img.shape)