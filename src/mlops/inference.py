import torch
from mlops.quantize import load_quantized_model

model = load_quantized_model(checkpoint_path="models/latest/best_model.pth", num_classes=43)

dummy_input = torch.randn(1, 3, 64, 64)

# Inference
with torch.inference_mode():
    logits = model(dummy_input)
    pred = logits.argmax(dim=1)

print(pred)
