from src.mlops.data import get_loaders
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        # --- Block 1 ---
        # Input: (Batch_Size, 3, 64, 64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Output after pool: (32, 32, 32)

        # --- Block 2 ---
        # Input: (32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Output after pool: (64, 16, 16)

        # --- Block 3 ---
        # Input: (64, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Output after pool: (128, 8, 8)

        # --- Classifier ---
        # The flatten size is: Channels * Height * Width
        # 128 channels * 8 pixels * 8 pixels = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten: (Batch_Size, 128, 8, 8) -> (Batch_Size, 8192)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
#     """Just a dummy model to show how to structure your code"""
#     def __init__(self):
#         super().__init__()
#         self.layer = nn.Linear(1, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.layer(x)

if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
    train_loader, test_loader = get_loaders(batch_size=64)


