import torch
import torch.nn as nn
import os
import logging
from hydra.utils import instantiate


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


# Initialize logger
log = logging.getLogger(__name__)

# UPDATED SIGNATURE: Added output_dir argument
def convert_to_onnx(checkpoint_path: str, output_filename: str = "model.onnx", output_dir: str = None):
    """
    Exports checkpoint to ONNX. 
    If output_dir is provided, saves the ONNX file there. 
    Otherwise, defaults to cfg.paths.models.
    """
    device = "cpu"
    
    log.info(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 1. Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'config' key.")
    
    # 2. Extract Config & Instantiate Model
    cfg = checkpoint["config"]
    log.info("Instantiating model from saved configuration...")
    
    model = instantiate(cfg.model)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # 3. Create Dummy Input
    height, width = cfg.image_size
    dummy_input = torch.randn(1, 3, height, width, device=device)
    
    # 4. Determine Output Directory (UPDATED LOGIC)
    if output_dir:
        # Use the specific parent directory passed from the BentoML script
        save_dir = output_dir
    else:
        # Fallback to the generic path in config if not specified
        save_dir = cfg.paths.models
        
    os.makedirs(save_dir, exist_ok=True)
    onnx_path = os.path.join(save_dir, output_filename)
    
    # 5. Export to ONNX
    log.info(f"Exporting ONNX model to: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    
    log.info("Export completed successfully.")
    return onnx_path

