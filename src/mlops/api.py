from contextlib import asynccontextmanager
from fastapi import UploadFile, File, FastAPI
from http import HTTPStatus
import cv2
import numpy as np
import torch
from torchvision import transforms
import os

# Hydra imports
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Import your model
from mlops.model import TinyCNN 

ml_models = {}
device = "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 1. Load Configuration with Hydra ---
    # Clear any previous Hydra instance to prevent errors during reloads
    GlobalHydra.instance().clear()
    
    # Initialize Hydra. 
    # config_path is relative to THIS file (src/mlops/api.py) => ../../configshydra
    with initialize(version_base="1.2", config_path="../../configshydra"):
        # Compose the configuration (loads config.yaml + defaults)
        cfg = compose(config_name="config")
    
    print(f"Hydra config loaded. Project: {cfg.wandb.project_name}")

    # --- 2. Extract Parameters ---
    # Get num_classes directly from config
    num_classes = cfg.data.num_classes
    
    # Construct the model path. 
    # train.py saves the latest model to: cfg.paths.models / "latest" / "best_model.pth"
    # We use to_absolute_path to resolve relative paths correctly based on where you run uvicorn
    models_dir = hydra.utils.to_absolute_path(cfg.paths.models)
    model_path = os.path.join(models_dir, "latest", "best_model.pth")

    print(f"Loading model from {model_path}...")

    # --- 3. Load Model ---
    model = TinyCNN(num_classes=num_classes).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle the checkpoint dictionary structure
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        ml_models["tiny_cnn"] = model
        print(f"Model loaded successfully with {num_classes} classes!")
    else:
        # Fail gracefully or raise an error depending on your needs
        print(f"CRITICAL WARNING: Model file not found at {model_path}")
        print("Inference will fail until the model is fixed.")

    yield 
    
    # --- Cleanup ---
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...)):
    if "tiny_cnn" not in ml_models:
        return {
            "message": "Model not loaded",
            "status-code": HTTPStatus.INTERNAL_SERVER_ERROR.value
        }
    
    model = ml_models["tiny_cnn"]

    # ... (Rest of your preprocessing and inference code remains the same) ...
    content = await data.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Note: You could also load image_size from cfg.image_size if you wanted!
    res = cv2.resize(img_rgb, (64, 64))
    
    input_tensor = transforms.ToTensor()(res)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_class = probabilities.topk(1, dim=1)

    return {
        "filename": data.filename,
        "predicted_class": top_class.item(),
        "probability": top_prob.item(),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK.value,
    }