import os
import time
import torch

from mlops.inference import load_inference_model


def benchmark(model: torch.nn.Module, x: torch.Tensor, n: int = 100) -> float:
    """
    Measure average inference latency in seconds.
    """
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
    # -------------------------
    # Configuration
    # -------------------------
    checkpoint_path = "models/latest/best_model.pth"
    num_classes = 43
    input_shape = (1, 3, 64, 64)
    n_runs = 100
    prune_amount = 0.3

    os.makedirs("models/latest", exist_ok=True)


    # Dummy input
    x = torch.randn(*input_shape)

    results = {}


    # 1. Float model
    float_model = load_inference_model(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
    )

    float_time = benchmark(float_model, x, n_runs)

    torch.save(float_model.state_dict(), "models/latest/float_tmp.pth")
    float_size = get_size_mb("models/latest/float_tmp.pth")
    os.remove("models/latest/float_tmp.pth")

    results["float"] = (float_time, float_size)


    # 2. Pruned model
    pruned_model = load_inference_model(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        prune_amount=prune_amount,
    )

    pruned_time = benchmark(pruned_model, x, n_runs)

    pruned_path = "models/latest/pruned_model.pth"
    torch.save(pruned_model.state_dict(), pruned_path)
    pruned_size = get_size_mb(pruned_path)

    results["pruned"] = (pruned_time, pruned_size)


    # 3. Quantized model
    quant_model = load_inference_model(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        quantized=True,
    )

    quant_time = benchmark(quant_model, x, n_runs)

    quant_path = "models/latest/quantized_model.pth"
    torch.save(quant_model.state_dict(), quant_path)
    quant_size = get_size_mb(quant_path)

    results["quantized"] = (quant_time, quant_size)

    # -------------------------
    # 4. Pruned + Quantized model
    # -------------------------
    pq_model = load_inference_model(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        prune_amount=prune_amount,
        quantized=True,
    )

    pq_time = benchmark(pq_model, x, n_runs)

    pq_path = "models/latest/pruned_quantized_model.pth"
    torch.save(pq_model.state_dict(), pq_path)
    pq_size = get_size_mb(pq_path)

    results["pruned+quantized"] = (pq_time, pq_size)


    # Print results
    print("\n=========   Inference Results   ==========")

    for name, (latency, size) in results.items():
        print(f"{name:>18}: {latency * 1000:7.2f} ms | {size:6.2f} MB")

    print("\n========= Relative Improvements =========")

    base_time, base_size = results["float"]

    for name, (latency, size) in results.items():
        speedup = base_time / latency
        compression = base_size / size
        print(f"{name:>18}: {speedup:5.2f}x faster | {compression:5.2f}x smaller")


if __name__ == "__main__":
    main()
