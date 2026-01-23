from __future__ import annotations
import bentoml
from PIL import Image
import numpy as np
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from onnxruntime import InferenceSession
from pathlib import Path
import torch.nn.functional as F
import torch
# Import your helper function
from mlops.export_onnx import convert_to_onnx

@bentoml.service
class ImageClassifierService:
    """Image classifier service using ONNX model."""

    def __init__(self) -> None:
        # 1. Initialize Hydra
        GlobalHydra.instance().clear()
        
        # config_path is relative to THIS file (src/mlops/)
        with initialize(version_base="1.2", config_path="../../configshydra"):
            self.cfg = compose(config_name="config")

        # 2. Resolve Paths
        ckpt_path = Path(hydra.utils.to_absolute_path(self.cfg.ckpt_path))
        models_dir = ckpt_path.parent
        onnx_filename = f"{ckpt_path.stem}.onnx"
        onnx_path = models_dir / onnx_filename
        
        # 3. Check/Convert Model
        if not onnx_path.exists():
            print(f"ONNX model not found at {onnx_path}. Converting...")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            convert_to_onnx(str(ckpt_path), onnx_filename, output_dir=str(models_dir))
            
        self.model = InferenceSession(str(onnx_path))

    # UPDATED: Accept PIL Image directly
    @bentoml.api
    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Predict the class of the input image.
        Auto-resizes image to match model requirements.
        """
        # 1. Resize Image
        # We use the size defined in your config (e.g., [64, 64])
        # Note: cfg.image_size is usually [Height, Width], PIL uses (Width, Height)
        target_height, target_width = self.cfg.image_size
        image = image.resize((target_width, target_height))
        
        # 2. Convert to Numpy & Normalize
        # Convert PIL image to numpy array (H, W, C)
        img_arr = np.array(image).astype(np.float32)
        
        # Transpose to (C, H, W) for PyTorch/ONNX
        img_arr = np.transpose(img_arr, (2, 0, 1))
        
        # Add batch dimension: (1, C, H, W)
        img_arr = np.expand_dims(img_arr, axis=0)

        raw_output = self.model.run(None, {"input": img_arr})
        tensor_output = torch.from_numpy(raw_output[0])

        # 3. Apply Softmax
        probabilities = F.softmax(tensor_output, dim=1)
        
        # Convert back to numpy to return (optional, BentoML handles tensors too)
        return probabilities.numpy()[0]