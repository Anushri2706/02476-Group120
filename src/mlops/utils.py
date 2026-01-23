import torch
import torch.nn as nn


def check_prune_level(module: nn.Module, name: str = ""):
    """
    Print sparsity (percentage of zero weights) for a module.
    """
    if not hasattr(module, "weight"):
        print(f"Module {name} has no weight parameter")
        return

    weight = module.weight.data
    zeros = torch.sum(weight == 0).item()
    total = weight.numel()

    sparsity = 100.0 * zeros / total
    print(f"Sparsity of {name or module.__class__.__name__}: {sparsity:.2f}%")
