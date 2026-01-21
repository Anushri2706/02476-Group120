from contextlib import asynccontextmanager
from fastapi import UploadFile, File, FastAPI, Request
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
from hydra.utils import instantiate
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

    
    
    # Construct the model path. 
    # train.py saves the latest model to: cfg.paths.models / "latest" / "best_model.pth"
    # We use to_absolute_path to resolve relative paths correctly based on where you run uvicorn
    model_path = hydra.utils.to_absolute_path(cfg.ckpt_path)
    print(f"Attempting to load model from {model_path}...")

    print(f"Loading model from {model_path}...")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract the Saved Config (architecture definition)
        saved_cfg = checkpoint['config']
        
        # Instantiate dynamically
        model = instantiate(saved_cfg.model).to(device)
        
        # Load Weights
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        # Attach to App State
        app.state.model = model
        app.state.config = saved_cfg 
        
        print(f"Model <{model.__class__.__name__}> loaded successfully!")
    else:
        app.state.model = None
        print(f"CRITICAL WARNING: Model file not found at {model_path}")

    yield 
    
    app.state.model = None

app = FastAPI(lifespan=lifespan)

@app.post("/cv_model/")
async def cv_model(request: Request, data: UploadFile = File(...)):
    # --- 4. Check if Model is Loaded ---
    if not hasattr(request.app.state, "model") or request.app.state.model is None:
        return {
            "message": "Model not loaded. Check server logs.",
            "status-code": HTTPStatus.INTERNAL_SERVER_ERROR.value
        }
    
    model = request.app.state.model
    # Use the saved config to ensure image size matches what the model expects
    cfg = request.app.state.config

    # --- 5. Inference ---
    content = await data.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use config-defined image size
    target_size = tuple(cfg.image_size)
    res = cv2.resize(img_rgb, target_size)
    
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