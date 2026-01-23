import torch
import torch.nn.utils.prune as prune
from .model import TinyCNN


def load_pruned_model(checkpoint_path: str, num_classes: int, amount: float = 0.3):
    """
    Load TinyCNN and apply unstructured magnitude pruning.
    Args:
        amount: fraction of weights to prune
    """

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = TinyCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Apply pruning to Linear layers
    prune.l1_unstructured(model.fc1, name="weight", amount=amount)
    prune.l1_unstructured(model.fc2, name="weight", amount=amount)

    # IMPORTANT: make pruning permanent
    prune.remove(model.fc1, "weight")
    prune.remove(model.fc2, "weight")

    return model
