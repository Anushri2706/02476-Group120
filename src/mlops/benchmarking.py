import os
import time
import torch

from mlops.model import TinyCNN
from mlops.quantize import load_quantized_model


def benchmark(model: torch.nn.Module, x: torch.Tensor, n: int = 100) -> float:
    """Return average inference time per forward pass (seconds)."""
    model.eval()
    with torch.inference_mode():
        # Warm-up
        for _ in range(10):
            _ = model(x)

        start = time.time()
        for _ in range(n):
            _ = model(x)
        end = time.time()

    return (end - start) / n


def get_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


def main():
    # Configuration
    num_classes = 43
    input_shape = (1, 3, 64, 64)
    n_runs = 100

    float_ckpt_path = "models/latest/best_model.pth"
    quant_ckpt_dir = "models/quantized"
    quant_ckpt_path = os.path.join(quant_ckpt_dir, "quantized_model.pth")

    os.makedirs(quant_ckpt_dir, exist_ok=True)

    # Load FLOAT model
    float_model = TinyCNN(num_classes=num_classes)
    checkpoint = torch.load(float_ckpt_path, map_location="cpu")
    float_model.load_state_dict(checkpoint["state_dict"])
    float_model.eval()

    # Load QUANTIZED model
    quant_model = load_quantized_model(
        checkpoint_path=float_ckpt_path,
        num_classes=num_classes,
    )

    # Save quantized model
    torch.save(quant_model.state_dict(), quant_ckpt_path)

    # Dummy input
    x = torch.randn(*input_shape)

    # Benchmarking
    float_time = benchmark(float_model, x, n=n_runs)
    quant_time = benchmark(quant_model, x, n=n_runs)


    torch.save(float_model.state_dict(), "models/latest/float_tmp.pth")

    float_size = get_size_mb("models/latest/float_tmp.pth")
    quant_size = get_size_mb(quant_ckpt_path)

    os.remove("models/latest/float_tmp.pth")

    print("=== Inference Benchmark ===")
    print(f"Float latency:     {float_time * 1000:.2f} ms")
    print(f"Quantized latency: {quant_time * 1000:.2f} ms")
    print()
    print("=== Model Size ===")
    print(f"Float model:     {float_size:.2f} MB")
    print(f"Quantized model: {quant_size:.2f} MB")
    print(f"Size reduction:  {float_size / quant_size:.2f}×")


if __name__ == "__main__":
    main()
