import torch
from torch.quantization import quantize_dynamic
from .model import TinyCNN  # adjust import if needed


def load_quantized_model(checkpoint_path: str, num_classes: int, device: str = "cpu"):
    """
    Load TinyCNN model and apply dynamic quantization
    """

    # 1. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Recreate model architecture
    model = TinyCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 3. Apply dynamic quantization (Linear layers only)
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    return quantized_model
