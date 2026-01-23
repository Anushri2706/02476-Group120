import torch
from torch import nn
from torch.quantization import quantize_dynamic
import torch.nn.utils.prune as prune

from .model import TinyCNN


def load_inference_model(
    checkpoint_path: str,
    num_classes: int,
    prune_amount: float | None = None,
    quantized: bool = False,
) -> nn.Module:
    """
    Load TinyCNN for inference with optional pruning and quantization.
    Return model ready for inference on CPU.

    Args:
        checkpoint_path: Path to trained model checkpoint (.pth)
        num_classes: Number of output classes
        prune_amount: Fraction of weights to prune (e.g. 0.3), or None
        quantized: Whether to apply dynamic quantization

    """

    # Load base model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = TinyCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()


    # Apply pruning if requested
    if prune_amount is not None:

        prune.l1_unstructured(model.fc1, name="weight", amount=prune_amount)
        prune.l1_unstructured(model.fc2, name="weight", amount=prune_amount)

        # Make pruning permanent
        prune.remove(model.fc1, "weight")
        prune.remove(model.fc2, "weight")

    # Apply quantization if requested
    if quantized:
        model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    return model


def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Run inference on a batch of inputs and returns model logits.

    Args:
        model: Inference-ready model
        x: Input tensor

    """
    with torch.inference_mode():
        return model(x)
